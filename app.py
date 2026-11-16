import os
import io
import json
import time
from typing import Optional

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(
    page_title="AcneClear AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Configuration -----------------------------
APP_TITLE = "AcneClear AI"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def get_api_key() -> str:
    """Read the Gemini key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")

def get_gemini_client():
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml or your deployment secrets."
        )
    return genai.Client(api_key=key)

def get_gemini_model() -> str:
    try:
        configured = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        configured = ""
    return configured or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def safe_generate(prompt: str) -> str:
    """Generate text with Gemini and return a useful error instead of crashing."""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.45,
            max_output_tokens=1800,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return text.strip()

# ----------------------------- Image processing -----------------------------
def enhance_photo(uploaded_file) -> tuple[Image.Image, str]:
    """
    Safe image enhancement pipeline.
    Uses GFPGAN if it has been installed and its weights are available.
    Otherwise uses a conservative OpenCV enhancement fallback.
    """
    try:
        original = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open this image: {exc}") from exc

    # Prevent unexpectedly huge images from exhausting memory.
    original.thumbnail((1600, 1600))
    rgb = np.array(original)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Optional GFPGAN path. It is not required for the app to run.
    try:
        from gfpgan import GFPGANer

        model_path = os.getenv("GFPGAN_MODEL_PATH", "weights/GFPGANv1.3.pth")
        if os.path.isfile(model_path):
            restorer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            _, _, restored = restorer.enhance(
                bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
            if restored is not None:
                restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
                return Image.fromarray(restored_rgb), "GFPGAN face restoration"
    except Exception:
        # Keep the app usable if the optional model or its dependencies fail.
        pass

    # Conservative fallback: mild denoising and subtle contrast correction.
    denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_bgr = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb), "OpenCV gentle enhancement"

def image_to_bytes(image: Image.Image, fmt="PNG") -> bytes:
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()

# ----------------------------- Prompt builders -----------------------------
def build_skincare_prompt(profile: dict) -> str:
    return f"""
You are a cautious skincare education assistant. Create a practical, beginner-friendly
skincare guidance report. This is educational information, not a diagnosis or medical
treatment. Do not claim that a photo has diagnosed the user's skin type or acne.

User-provided profile:
- Skin type (self-reported): {profile['skin_type']}
- Main concerns (self-reported): {', '.join(profile['concerns'])}
- Sensitivity: {profile['sensitivity']}
- Current routine: {profile['routine'] or 'Not provided'}
- Product budget: {profile['skincare_budget']}

Write the response with these headings:
1. Understanding the stated concerns
2. Simple morning routine
3. Simple evening routine
4. Ingredients to learn about (include patch-test advice)
5. Habits that may help
6. When to consult a dermatologist

Avoid prescribing medication, diagnosing conditions, promising results, or recommending
harsh home remedies. If symptoms are painful, severe, scarring, infected, or persistent,
recommend consulting a qualified dermatologist. Keep the advice suitable for an Indian user.
"""

def build_diet_prompt(profile: dict) -> str:
    return f"""
You are a general nutrition education assistant. Create a balanced, realistic meal plan.
Do not claim that food cures acne or replaces medical care. Do not prescribe supplements.

User-provided preferences:
- Food preference: {profile['food_preference']}
- Cuisine: {profile['cuisine']}
- Daily food budget: {profile['food_budget']}
- Meals per day: {profile['meals_per_day']}
- Allergies / foods to avoid: {profile['avoid_foods'] or 'Not provided'}
- Other preferences: {profile['food_notes'] or 'Not provided'}

Create a practical 1-day Indian meal plan with breakfast, lunch, evening snack, and dinner.
For each meal, give 2 options, simple portion guidance, and affordable substitutions.
Respect all stated restrictions. Include hydration and balanced-eating notes. Mention that
individual nutrition needs vary and severe allergies require careful label checking and
professional advice. Do not make acne-treatment claims.
"""

def build_chat_prompt(message: str, history: list, profile: Optional[dict]) -> str:
    context = ""
    if profile:
        context = f"""
User's optional saved preferences:
Skin type: {profile.get('skin_type', 'Not provided')}
Concerns: {', '.join(profile.get('concerns', []))}
Sensitivity: {profile.get('sensitivity', 'Not provided')}
Food preference: {profile.get('food_preference', 'Not provided')}
Foods to avoid: {profile.get('avoid_foods', 'Not provided')}
"""
    recent = history[-10:]
    conversation = "\n".join(
        f"{item['role'].upper()}: {item['content']}" for item in recent
    )
    return f"""
You are AcneClear AI, a friendly skincare and general-wellness information assistant.
Give clear, cautious, non-judgmental educational information. Do not diagnose from images,
prescribe medication, promise acne cures, or replace a dermatologist or dietitian.
For severe, painful, scarring, infected, or persistent skin concerns, suggest a dermatologist.
Do not recommend supplements or restrictive diets as acne cures.
{context}
Recent conversation:
{conversation}

USER: {message}
ASSISTANT:
"""

# ----------------------------- UI -----------------------------
st.title("✨ AcneClear AI")
st.caption("A connected photo-enhancement and skin-wellness guidance experience")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["Personalized Skin Report", "AI Chatbot", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Educational guidance only. Not a medical diagnosis or treatment.")

if "profile" not in st.session_state:
    st.session_state.profile = None
if "report" not in st.session_state:
    st.session_state.report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if page == "Personalized Skin Report":
    st.subheader("Your connected skin-wellness report")
    st.write(
        "Upload one clear photo and enter your preferences once. "
        "AcneClear will process the image and generate skincare guidance and a meal plan."
    )

    with st.form("skin_report_form"):
        uploaded = st.file_uploader(
            "Upload a clear facial photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a well-lit photo. The app does not identify you or diagnose skin conditions.",
        )

        st.markdown("#### Skin preferences")
        c1, c2 = st.columns(2)
        with c1:
            skin_type = st.selectbox(
                "Your skin type (self-reported)",
                ["Not sure", "Oily", "Dry", "Combination", "Normal", "Sensitive"],
            )
            sensitivity = st.selectbox(
                "Skin sensitivity",
                ["Not sure", "Low", "Moderate", "High"],
            )
            skincare_budget = st.selectbox(
                "Skincare budget",
                ["Low budget", "Moderate budget", "Flexible budget"],
            )
        with c2:
            concerns = st.multiselect(
                "Main concerns",
                ["Acne / pimples", "Dark spots", "Oiliness", "Dryness",
                 "Redness", "Uneven texture", "Blackheads", "General skin care"],
            )
            routine = st.text_input(
                "Current skincare routine (optional)",
                placeholder="Example: face wash and moisturizer",
            )

        st.markdown("#### Food preferences")
        c3, c4 = st.columns(2)
        with c3:
            food_preference = st.selectbox(
                "Food preference",
                ["Vegetarian", "Eggetarian", "Non-vegetarian", "Vegan"],
            )
            cuisine = st.selectbox(
                "Preferred meal style",
                ["South Indian", "North Indian", "Mixed Indian", "No preference"],
            )
            food_budget = st.selectbox(
                "Daily food budget",
                ["Below ₹100", "₹100–₹200", "₹200–₹350", "Flexible"],
            )
        with c4:
            meals_per_day = st.selectbox("Meals per day", ["3", "4", "5"])
            avoid_foods = st.text_input(
                "Allergies or foods to avoid (optional)",
                placeholder="Example: peanuts, milk",
            )
            food_notes = st.text_input(
                "Other food preferences (optional)",
                placeholder="Example: hostel-friendly, no cooking",
            )

        submitted = st.form_submit_button(
            "Generate my complete report", type="primary", use_container_width=True
        )

    if submitted:
        if uploaded is None:
            st.error("Please upload a photo before generating your report.")
        elif not concerns:
            st.error("Please select at least one skin concern, or choose General skin care.")
        else:
            profile = {
                "skin_type": skin_type,
                "concerns": concerns,
                "sensitivity": sensitivity,
                "routine": routine.strip(),
                "skincare_budget": skincare_budget,
                "food_preference": food_preference,
                "cuisine": cuisine,
                "food_budget": food_budget,
                "meals_per_day": meals_per_day,
                "avoid_foods": avoid_foods.strip(),
                "food_notes": food_notes.strip(),
            }
            st.session_state.profile = profile
            st.session_state.report = None

            progress = st.progress(0, text="Starting your report...")
            status = st.empty()
            results = {"profile": profile}

            # Stage 1: image processing
            try:
                status.info("Step 1 of 3: Processing your photo...")
                original_bytes = uploaded.getvalue()
                enhanced, method = enhance_photo(uploaded)
                results["original_image"] = original_bytes
                results["enhanced_image"] = image_to_bytes(enhanced)
                results["enhancement_method"] = method
                progress.progress(33, text="Photo processing complete")
            except Exception as exc:
                results["image_error"] = str(exc)
                progress.progress(33, text="Photo processing had a problem")

            # Stage 2: skincare guidance
            try:
                status.info("Step 2 of 3: Generating your skincare guidance...")
                results["skincare"] = safe_generate(build_skincare_prompt(profile))
                progress.progress(66, text="Skincare guidance complete")
            except Exception as exc:
                results["skincare_error"] = str(exc)
                progress.progress(66, text="Skincare generation had a problem")

            # Stage 3: diet plan
            try:
                status.info("Step 3 of 3: Creating your meal plan...")
                results["diet"] = safe_generate(build_diet_prompt(profile))
                progress.progress(100, text="Report generation finished")
            except Exception as exc:
                results["diet_error"] = str(exc)
                progress.progress(100, text="Diet-plan generation had a problem")

            st.session_state.report = results
            status.success("Your report process has finished. Review the results below.")

    report = st.session_state.report
    if report:
        st.divider()
        st.header("Your personalized report")

        if "enhanced_image" in report:
            left, right = st.columns(2)
            with left:
                st.markdown("**Uploaded photo**")
                # Original image is displayed only for this active upload/session.
                st.image(report["original_image"], caption="Original photo",
                         use_container_width=True)
            with right:
                st.markdown("**Enhanced photo**")
                st.image(report["enhanced_image"], caption=report["enhancement_method"],
                         use_container_width=True)
                st.download_button(
                    "Download enhanced photo",
                    data=report["enhanced_image"],
                    file_name="acneclear_enhanced.png",
                    mime="image/png",
                )
        else:
            st.warning("Photo enhancement was not completed.")
            if report.get("image_error"):
                st.caption(report["image_error"])

        st.subheader("Skincare guidance")
        if report.get("skincare"):
            st.markdown(report["skincare"])
        else:
            st.error("Skincare guidance could not be generated.")
            st.caption(report.get("skincare_error", "Unknown error"))

        st.subheader("Your meal plan")
        if report.get("diet"):
            st.markdown(report["diet"])
        else:
            st.error("The meal plan could not be generated.")
            st.caption(report.get("diet_error", "Unknown error"))

        st.caption(
            "The image enhancement is cosmetic image processing. It does not diagnose "
            "acne or prove that a skin condition has improved."
        )

elif page == "AI Chatbot":
    st.subheader("AcneClear AI Chatbot")
    st.write("Ask general questions about skincare routines, ingredients, or balanced meals.")
    use_profile = st.checkbox(
        "Use my preferences from the latest report",
        value=st.session_state.profile is not None,
        disabled=st.session_state.profile is None,
    )
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("Ask AcneClear AI...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    profile = st.session_state.profile if use_profile else None
                    answer = safe_generate(
                        build_chat_prompt(prompt, st.session_state.chat_history[:-1], profile)
                    )
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"I couldn't generate a response: {exc}"
                    st.error(answer)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.subheader("About AcneClear AI")
    st.markdown(
        """
        AcneClear AI combines:
        - **Photo enhancement:** GFPGAN if configured, otherwise gentle OpenCV processing.
        - **Skincare education:** Gemini creates general guidance from your self-reported preferences.
        - **Meal planning:** Gemini creates meal ideas using your food preferences and restrictions.
        - **Separate chatbot:** Ask follow-up questions independently.

        **Limitations:** A photo is not used to diagnose acne or infer food needs. The app does
        not prescribe treatment, and food plans are general suggestions rather than medical
        nutrition therapy. Consult a qualified professional for persistent or severe concerns.
        """
    )
