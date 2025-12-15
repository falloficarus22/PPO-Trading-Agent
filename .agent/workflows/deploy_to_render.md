---
description: How to deploy the PPO Trading Agent to the internet using Render.com
---

# Deploying PPO Trading Agent

This guide will help you deploy your trading bot to the internet so it can run 24/7. We will use **Render** (render.com) because it is easy, supports Docker, and has a free tier for web services.

## Prerequisites
- A GitHub account.
- The project pushed to a GitHub repository.

## Step 1: Push to GitHub
If you haven't already, push your code to a new GitHub repository.
```bash
git init
git add .
git commit -m "Initial commit for deployment"
# Create a repo on GitHub.com and follow the 'push an existing repository' instructions
```

## Step 2: Create a Render Account
1. Go to [dashboard.render.com](https://dashboard.render.com/).
2. Sign up/Log in using your GitHub account.

## Step 3: Create a New Web Service
1. Click the **"New +"** button and select **"Web Service"**.
2. Select **"Build and deploy from a Git repository"**.
3. Connect your GitHub account if asked, then select your `trader` repository.

## Step 4: Configure the Service
Render will detect the `Dockerfile`. Use the following settings:

- **Name**: `ppo-trader` (or any name you like)
- **Region**: Choose the one closest to you (e.g., Frankfurt, Oregon)
- **Branch**: `main` (or `master`)
- **Runtime**: **Docker** (This is important! Do not select Python)
- **Instance Type**: **Free** (Good for testing)

**Environment Variables:**
You need to set any secrets here if you have them (e.g., API keys). 
Since this is a paper trader, you might not have real API keys yet, but if you change `config.py` to use environment variables later, add them here.

## Step 5: Deploy
1. Click **"Create Web Service"**.
2. Render will start building your Docker image. This might take a few minutes as it downloads PyTorch.
3. Once built, it will deploy the service and give you a URL (e.g., `https://ppo-trader.onrender.com`).

## Step 6: Verify
1. Open the URL provided by Render.
2. You should see your dashboard!
3. Click "Start Live Trading".

**Note:** On the free tier, Render spins down services after inactivity. For a 24/7 trading bot, you will eventually need the **Starter** plan ($7/month) to prevent it from sleeping.
