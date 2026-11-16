# AcneClear AI

A Streamlit application with one connected workflow:
1. Upload a facial photo and enter preferences once.
2. Process the photo using optional GFPGAN or the OpenCV fallback.
3. Generate skincare education with Gemini.
4. Generate a meal plan with Gemini.
5. Use a separate AI chatbot page.

## 1. Create a Gemini API key
Create an API key in Google AI Studio. Keep it private. Do not commit it to GitHub.

## 2. Local setup (VS Code)

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` and add:
```toml
GEMINI_API_KEY = "PASTE_YOUR_KEY_HERE"
GEMINI_MODEL = "gemini-2.5-flash"
```

Run:
```bash
streamlit run app.py
```

## 3. Google Colab
Upload/extract this project in Colab. Install requirements:
```python
!pip install -r requirements.txt
```
Set the key securely in the Colab session (do not print it):
```python
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter Gemini API key: ")
```
For a temporary Colab preview, install and use a tunneling service only if you understand its access implications. For the final public app, deploy through Streamlit Community Cloud.

## 4. Streamlit Community Cloud
1. Push the project files to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. In app settings > Secrets, add:
```toml
GEMINI_API_KEY = "YOUR_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```
5. Deploy.

## Optional GFPGAN
The app works without GFPGAN using the OpenCV fallback. GFPGAN is optional because its dependencies and model weights can be platform-sensitive.

If you have a compatible GFPGAN installation and the `GFPGANv1.3.pth` weights file, place it at `weights/GFPGANv1.3.pth` or set `GFPGAN_MODEL_PATH` to the weights path. Install a compatible GFPGAN/BasicSR/Real-ESRGAN environment separately and test it locally before deployment.

## Important limitations
- OpenCV fallback provides gentle cosmetic image enhancement, not acne removal.
- GFPGAN is a general face-restoration model, not an acne-specific clinical model.
- The app does not diagnose skin conditions or prescribe treatment.
- Diet suggestions are general meal ideas, not a treatment for acne.
- Do not upload photos or personal details unless you are comfortable processing them with this app.
