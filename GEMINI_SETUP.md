# How to Get Your Free Gemini API Key

## Step 1: Visit Google AI Studio
Go to: https://makersuite.google.com/app/apikey

## Step 2: Sign in with Google Account
Use your Google account to sign in

## Step 3: Create API Key
1. Click "Create API Key"
2. Select "Create API key in new project" (or use existing project)
3. Copy the generated API key

## Step 4: Add to Your Project
1. Create a file named `.env` in your project root (d:\Climate_Intelligence_Project)
2. Add this line:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. Replace `your_actual_api_key_here` with the key you copied

## Step 5: Restart the App
```bash
streamlit run app.py
```

## ✨ Benefits of Gemini API:
- ✅ **FREE** tier with generous limits
- ⚡ **Instant responses** (no local hardware needed)
- 🎯 **Accurate AI summaries** of climate documents
- 🌐 **Cloud-based** - works on any computer

## Free Tier Limits:
- 60 requests per minute
- 1,500 requests per day
- More than enough for development and testing!
