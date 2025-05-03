import streamlit as st
import pandas as pd
import time
import uuid
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from supabase import create_client
import streamlit as st
import uuid
import requests     # ← needed for prefetching
import streamlit as st
import time
from datetime import datetime, timezone
from streamlit_cookies_manager import CookieManager

# This should be on top of your script
cookies = CookieManager(
    # This prefix will get added to all your cookie names.
    # This way you can run your app on Streamlit Cloud without cookie name clashes with other apps.
    prefix="DeepfakeSurvey/streamlit-cookies-manager/"
)


# ─── at the very top of your script, after imports ─────────────────────────────
# Make fonts & buttons a bit larger and center everything
st.markdown(
    """
    <style>
      /* Center the main app column and cap max-width */
      .stApp {
        max-width: 800px !important;
        margin: auto !important;
        padding: 2rem 1rem !important;
      }

      /* Tweak forms and images */
      form[role="form"],
      .stImage > div {
        background: var(--secondaryBackgroundColor) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
      }

      /* Hide the “placeholder” radio option label + its button */
      /* This hides the FIRST radio button inside every st.radio */
      .stRadio > div > label:nth-of-type(1),
      .stRadio > div > label:nth-of-type(1) ~ span[role="radio"] {
        display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)





# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

NUM_TRIALS    = 30 # 30 images per user
DISPLAY_TIME  = 5  # seconds per image

# ─── DATABASE SETUP ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_engine():
    db_url = st.secrets["database"]["url"]
    return create_engine(db_url, connect_args={"sslmode": "require"},
                        pool_size=15,
                            max_overflow=5,
                            pool_pre_ping=True)

engine = init_engine()

# Ensure responses table exists
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.responses (
            user_id   TEXT      NOT NULL,
            image_id  TEXT      NOT NULL,
            response  TEXT,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT responses_pkey PRIMARY KEY (user_id, image_id),
            CONSTRAINT responses_image_data_fkey
              FOREIGN KEY (image_id)
              REFERENCES public.image_data(image_id)
              ON UPDATE CASCADE
              ON DELETE RESTRICT
        );
    """))

# ─── METADATA LOADING ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_full_metadata_from_db():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT image_id,
                   dataset,
                   subset,
                   filename,
                   video,
                   label,
                   method 
            FROM public.image_data
        """)).all()
    df = pd.DataFrame(rows, columns=[
        "image_id","dataset","subset", "filename", "video",
        "label", "method"
    ])
    df["image_id"] = df["image_id"].astype(int)
    df["method"] = df["method"].replace("null", None)
    if len(df) < NUM_TRIALS:
        st.error(f"Need at least {NUM_TRIALS} images in your image_data table")
        st.stop()
    return df

full_metadata = load_full_metadata_from_db()



# 2) ask it to load existing cookies before you do anything else
if not cookies.ready():
    # early exit until cookies are loaded
    st.stop()

# 3) if there's no user_id yet, mint one and save it
if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()  # writes the Set-Cookie header back to the browser

user_id = cookies["user_id"]

# ─── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
if 'initialized' not in st.session_state:
    st.session_state.initialized    = True
    st.session_state.user_id        = user_id
    st.session_state.phase          = 'instruction'
    st.session_state.trial_index    = 0
    st.session_state.start_time     = None
    st.session_state.last_idx       = None           
    st.session_state.metadata       = None
    st.session_state.image_urls     = None
    st.session_state.image_bytes    = None
    st.session_state.responses      = []
if 'ready_for_response' not in st.session_state:
    st.session_state.ready_for_response = False


# ─── PREFETCH ALL 30 IMAGE BYTES (FROM 45 IMAGE POOL WHICH INCLUDES BACKUPS ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def prefetch_images(
    primary_urls: tuple[str, ...],
    backup_urls:  tuple[str, ...],
    target_count: int = NUM_TRIALS,
    max_retries:  int   = 3,
    backoff_factor: float = 0.5,
) -> dict[str, bytes]:
    """
    Fetch up to `target_count` images, first trying each URL in `primary_urls`.
    If any primary fails after max_retries, pull from backup_urls (in order)
    until we reach `target_count` successes or exhaust backups.

    Returns a dict {url: content} of length <= target_count.
    """
    import requests, time

    def try_fetch(url):
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                return r.content
            except Exception:
                wait = backoff_factor * (2 ** (attempt - 1))
                time.sleep(wait)
        return None

    images = {}
    used_backups = set()  # to avoid retrying the same backup twice

    # 1) Attempt primaries
    for url in primary_urls:
        if len(images) >= target_count:
            break
        content = try_fetch(url)
        if content is not None:
            images[url] = content

    # 2) Fill from backups if needed
    backup_iter = iter(backup_urls)
    while len(images) < target_count:
        try:
            b = next(backup_iter)
        except StopIteration:
            break  # no more backups
        if b in used_backups:
            continue
        used_backups.add(b)
        content = try_fetch(b)
        if content is not None:
            images[b] = content

    return images

# ─── SAMPLING ───────────────────────────────────────────────────────────────────
def sample_metadata(user_id: str, full_metadata: pd.DataFrame):
    """
    Draws exactly 30 images per session:
     -  3× FF++ real
     -  3× each FF++ fake method (5 methods)
     -  2× each of CelebDF-v1 real/fake
     -  2× each of CelebDF-v2 real/fake
     -  2× each of DFDC real/fake
    Filters out images the user has already seen (if possible),
    and falls back to replacement sampling if a bucket runs dry.
    """
     # how many primary vs backup
    PRIMARY, BACKUP = NUM_TRIALS, 15
    # 1) Which images has the user already labeled?
    with engine.connect() as conn:
        seen = {
            row[0]
            for row in conn.execute(
                text("SELECT image_id FROM public.responses WHERE user_id = :uid"),
                {"uid": user_id}
            )
        }

    # 2) Unseen pool (or full if < NUM_TRIALS remaining)
    unseen = full_metadata.loc[~full_metadata["image_id"].isin(seen)]
    pool   = unseen if len(unseen) >= PRIMARY else full_metadata.copy()

     # 3) Sample primary exactly PRIMARY using your bucket logic
    def sample_bucket(df_pool, filt, cnt):
        df = df_pool
        df = df[df["dataset"] == filt["dataset"]]
        df = df[df["label"]   == filt["label"]]
        if filt["method"] is None:
            df = df[df["method"].isnull()]
        else:
            df = df[df["method"] == filt["method"]]
        return df.sample(n=cnt, replace=(len(df) < cnt))

    # 3) Define buckets → (filter criteria, count)
    buckets = [
        # FF++ real
        ({"dataset":"FFPP_C23","label":"real","method":"youtube"}, 3),
        # FF++ fake subtypes
        ({"dataset":"FFPP_C23","label":"fake","method":"Deepfakes"},     3),
        ({"dataset":"FFPP_C23","label":"fake","method":"FaceSwap"},      3),
        ({"dataset":"FFPP_C23","label":"fake","method":"Face2Face"},     3),
        ({"dataset":"FFPP_C23","label":"fake","method":"FaceShifter"},   3),
        ({"dataset":"FFPP_C23","label":"fake","method":"NeuralTextures"},3),
        # CelebDF-v1
        ({"dataset":"CELEBDFV1_REAL","label":"real","method":None}, 2),
        ({"dataset":"CELEBDFV1_FAKE","label":"fake","method":"CELEBDFV1"}, 2),
        # CelebDF-v2
        ({"dataset":"CELEBDFV2_REAL","label":"real","method":None}, 2),
        ({"dataset":"CELEBDFV2_FAKE","label":"fake","method":"CELEBDFV2"}, 2),
        # DFDC
        ({"dataset":"DFDC_REAL","label":"real","method":None}, 2),
        ({"dataset":"DFDC_FAKE","label":"fake","method":"DFDC"}, 2),
    ]

    primary_samples = []
    for filt, cnt in buckets:
        primary_samples.append(sample_bucket(pool, filt, cnt))
    prim_df = pd.concat(primary_samples, axis=0)
    # if we've inadvertently drawn more or fewer than PRIMARY, resample/shrink
    if len(prim_df) > PRIMARY:
        prim_df = prim_df.sample(n=PRIMARY, replace=False)
    elif len(prim_df) < PRIMARY:
        extra = pool.drop(prim_df.index).sample(n=PRIMARY-len(prim_df), replace=True)
        prim_df = pd.concat([prim_df, extra], axis=0)

    # 4) Build backup pool *excluding* the primary picks
    remaining = full_metadata.drop(prim_df.index)
    if len(remaining) < BACKUP:
        # if too small, fall back to full metadata
        remaining = full_metadata.copy()
    back_df = remaining.sample(n=BACKUP, replace=(len(remaining) < BACKUP))

    # 5) concatenate: prim first, then backup
    sampled = pd.concat([prim_df, back_df], axis=0).reset_index(drop=True)
    return sampled


# ─── UI COMPONENTS ──────────────────────────────────────────────────────────────
def show_countdown(start_time):
    remaining = DISPLAY_TIME - int(time.time() - start_time)
    if remaining > 0:
        st.empty().markdown(f"**Time left:** {remaining}s")
        st.rerun()
    else:
        st.session_state.phase = 'get_response'


def show_instructions():
    box = st.container()
    with box:
        st.title("Deepfake Detection Test")
        st.write( f"""
            This test is designed to measure how well people can tell real and fake human faces apart. You will be shown {NUM_TRIALS} images, one at a time,
            for {DISPLAY_TIME} seconds each. After the image disappears, you’ll be asked to answer whether you believe the image was real or fake. You may take as long as you need to answer.
            The results of this test will be used for research purposes only. When you’re ready, click **Begin Test**.
            """)
        # -- once, generate your metadata and image‐bytes in the background --
        # If we haven’t sampled & prefetched yet, do it under a spinner
        if st.session_state.metadata is None:
            with st.spinner("Getting everything ready…"):
                # 1) Sample metadata
                st.session_state.metadata = sample_metadata(
                    st.session_state.user_id,
                    full_metadata
                )
                # 2) Build all S3 URLs
                BASE = "https://rpkeffblqusmqojobkjq.supabase.co/storage/v1/object/public/deepfakesurvey"
                urls = tuple(f"{BASE}/{fn}"
                            for fn in st.session_state.metadata["filename"])
                
                primary_urls = urls[:NUM_TRIALS]
                backup_urls  = urls[NUM_TRIALS:]

                # prefetch images (use backup if any one fails):
                st.session_state.image_bytes = prefetch_images(primary_urls, backup_urls)
                st.session_state.image_urls  = list(st.session_state.image_bytes.keys())[:NUM_TRIALS]

            # A tiny delay so the spinner actually shows up
            time.sleep(0.1)

        if st.button("Begin Test"):
            box.empty()
            st.session_state.trial_index         = 0
            st.session_state.phase               = "show_image"
            st.session_state.start_time          = time.time()
            st.session_state.last_idx            = -1  # ensure first frame sets timer
            st.session_state.responses           = []
            st.session_state.ready_for_response  = False
            st.rerun()


def show_image():
    idx = st.session_state.trial_index
    md  = st.session_state.metadata

    # initialize timer once
    if st.session_state.last_idx != idx:
        st.session_state.start_time = time.time()
        st.session_state.last_idx   = idx

    st.image(
        st.session_state.image_bytes[st.session_state.image_urls[idx]],
        use_container_width=True
    )

    st.markdown(f"### {idx+1} / {NUM_TRIALS}")
    # progress bar with label
    # countdown
    elapsed   = time.time() - st.session_state.start_time
    remaining = max(0, DISPLAY_TIME - int(elapsed))
    st.markdown(f"**Time left:** {remaining}s")

    # rerun logic
    if remaining > 0:
        time.sleep(1)
        st.rerun()
    else:
        if not st.session_state.ready_for_response:
            st.session_state.ready_for_response = True
            st.rerun()


def get_response():
    box = st.container()
    with box:
        idx = st.session_state.trial_index
        row = st.session_state.metadata.iloc[idx]

        st.markdown(f"## Image {idx+1} of {NUM_TRIALS}")
        st.write("")  # breathing room

        # 1) Render the radio *outside* the form
        options = ["","🟢 Real", "🔴 Fake"]

        choice = st.radio(
            "Your Answer",
            options,
            index=0,                # no default
            key=f"radio_{idx}",
            horizontal=True,
        )

        # only enable once they pick one of the real options
        can_submit = choice in options[1:]

        with st.form(key=f"resp_form_{idx}", clear_on_submit=True, border=False):
            submitted = st.form_submit_button(
                "Submit",
                use_container_width=True,
                disabled=not can_submit
            )
            if not can_submit:
                st.caption("Please select Real or Fake to continue.")

            if submitted:
                box.empty() # remove the form
                # write to DB
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO public.responses (user_id,image_id,response,timestamp)
                        VALUES(:uid,:iid,:resp,:ts)
                        ON CONFLICT DO NOTHING
                    """), {
                        "uid": user_id,
                        "iid": int(row["image_id"]),
                        "resp": choice[-4:].lower(),
                        "ts": datetime.now(timezone.utc),
                    })

                # record & advance
                st.session_state.responses.append({
                    "gt": row["label"],
                    "resp": choice[-4:].lower()
                })
                if idx + 1 < NUM_TRIALS:
                    st.session_state.trial_index += 1
                    st.session_state.phase = "show_image"
                else:
                    st.session_state.phase = "finished"

                # reset for next round
                st.session_state.ready_for_response = False
                st.rerun()




import random

def funny_message(correct: int, total: int) -> str:
    pct = correct / total
    # Define buckets of humor
    messages = [
        (0.0, 0.2, [
            "Oof… Did you even look at the pictures? 🤔",
            "My grandma does better blindfolded! 🧓😜",
            "Congratulations, you’ve discovered a new level of confusion.",
            "Was it really THAT hard?",
            "At least you’re consistent… consistently wrong!"
        ]),
        (0.2, 0.5, [
            "Not your best day, huh? It's okay, we've all been there.",
            "Better luck next time—maybe try coffee first ☕",
            "I don't know what to say, maybe you forgot your glasses?"
            ,

        ]),
        (0.5, 0.8, [
            "Pretty solid! You’ve got a face-spotting future. 🔮",
            "Nice work—your spidey-sense is tingling! 🕷️",
            "You’re almost a deepfake detective 🕵️‍♂️",
            "Sherlock Holmes would be proud!"
        ]),
        (0.8, 0.9, [
            "Whoa, are you a human or a machine? 🤖",
            "Absolute legend! 20/20 vision just like Superman! 🦸‍♂️",
            "I bow to your uncanny powers of observation.",
            "Quite the outlier, bravo!",
            "Was it really THAT easy?"
        ]),
        (0.92, 1.0, [
            "Hey!, quit being so good, you'll skew my data!",
            "I really hope you're not cheating."

        ])
    ]
    for low, high, msgs in messages:
        if low <= pct < high or (pct == 1.0 and high == 1.0):
            return random.choice(msgs)
    return "Well… that was unexpected! 🎉"


def show_finished():
    # compute score
    total   = len(st.session_state.responses)
    correct = sum(
        1 for r in st.session_state.responses
        if (r["gt"] and r["resp"] == "real") or
           (not r["gt"] and r["resp"] == "fake")
    )
    
    st.header("✅ Test Complete!")
    st.subheader(f"Your score: **{correct}/{total}**")
    st.write("")
    st.write(funny_message(correct, total))
    st.write("")
    if st.button("Retake Test"):
        for k in ("phase","trial_index","start_time","last_idx","responses"):
            st.session_state.pop(k, None)
        st.session_state.phase = "instruction"
        st.rerun()

    

# ─── ROUTER ─────────────────────────────────────────────────────────────────────
# ─── ROUTER ───────────────────────────────────────────────────────────────
if st.session_state.phase == 'instruction':
    show_instructions()
elif st.session_state.phase == 'show_image':
    if st.session_state.ready_for_response:
        st.session_state.phase = 'get_response'
        st.rerun()
    else:
        show_image()
elif st.session_state.phase == 'get_response':
    get_response()
elif st.session_state.phase == 'finished':
    show_finished()
else:
    st.error("Unexpected error occurred. Please refresh.")

