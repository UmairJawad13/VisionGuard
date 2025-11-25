# GitHub & Streamlit Deployment Guide

## Step 1: Initialize Git Repository

```powershell
# Navigate to your project directory
cd "C:\Users\user\Documents\UOW\Image Processing & Computer vision\Assignmnet 2\IPCV A2"

# Initialize Git
git init

# Add all files (respects .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: VisionGuard AI Vision Assistant"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com
2. Click the **"+"** icon (top-right) → **"New repository"**
3. Repository name: `visionguard` (or your choice)
4. Description: `AI-powered vision assistance for visually impaired users using YOLOv8 and LLaVA`
5. Choose **Public** or **Private**
6. **DO NOT** initialize with README (you already have one)
7. Click **"Create repository"**

## Step 3: Push to GitHub

```powershell
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/visionguard.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**Enter your GitHub credentials when prompted.**

## Step 4: Deploy to Streamlit Cloud

### A. Sign up for Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub

### B. Deploy Your App
1. Click **"New app"** (top-right)
2. Fill in the form:
   - **Repository:** `YOUR_USERNAME/visionguard`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Click **"Advanced settings"** (optional):
   - **Python version:** 3.11
   - **Requirements file:** `streamlit_requirements.txt`
4. Click **"Deploy!"**

### C. Wait for Deployment
- Streamlit will install dependencies (takes 5-10 minutes first time)
- Watch the build logs for any errors
- Your app will be live at: `https://YOUR_USERNAME-visionguard-streamlit-app-xxxxx.streamlit.app`

## Step 5: Update Your README

Once deployed, update the README with your actual app URL:

```powershell
# Edit README.md and replace the placeholder URL
# Then commit and push:
git add README.md
git commit -m "Update live demo URL"
git push
```

## Troubleshooting

### Git Authentication Issues
If you have 2FA enabled on GitHub, use a **Personal Access Token**:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Streamlit Deployment Errors

**Error: "Requirements file not found"**
- Solution: Make sure `streamlit_requirements.txt` is in the root directory
- Set "Requirements file" to `streamlit_requirements.txt` in Advanced settings

**Error: "ModuleNotFoundError: torch"**
- Solution: PyTorch is large (~2GB). Streamlit Cloud may timeout.
- Alternative: Use `torch` CPU version in requirements:
  ```
  torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  ```

**Error: "Out of memory"**
- Solution: YOLOv8 models are memory-intensive
- Use smallest model: `yolov8n.pt` (already set as default)
- Consider disabling features or using lighter alternatives

**App is slow:**
- Normal for free tier (limited CPU/RAM)
- Consider upgrading to Streamlit Cloud Pro for better performance

## Testing Locally First

Before deploying, test the Streamlit app locally:

```powershell
# Install Streamlit dependencies
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

Your browser should open automatically to `http://localhost:8501`

## Updating Your Deployed App

Whenever you make changes:

```powershell
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will **automatically redeploy** when it detects changes on GitHub!

## Sharing Your App

Once deployed, share the URL:
- Add it to your GitHub repository description
- Include it in your assignment report
- Share with friends/classmates

---

## Quick Reference Commands

```powershell
# Check Git status
git status

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# View remote URL
git remote -v

# Pull latest changes
git pull origin main

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

---

**Need help?** Check the Streamlit Community forums: https://discuss.streamlit.io
