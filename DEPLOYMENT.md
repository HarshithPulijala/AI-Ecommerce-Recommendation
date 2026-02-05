# Deployment Guide - AI Product Recommendation Web App

## 📋 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Virtual Environment activated
- All dependencies installed (`pip install -r requirements.txt`)

### Running Locally

**Terminal 1 - Start Flask Server:**
```bash
cd c:\Users\Anwith Pulijala\Desktop\Harshith.in\AI -ecommerce
$env:FLASK_APP="src.app"
python -m flask run --port 5000
```

**Terminal 2 - Test the Application:**
```bash
cd c:\Users\Anwith Pulijala\Desktop\Harshith.in\AI -ecommerce
python test_webapp.py
```

**Browser - Access Web Interface:**
```
http://localhost:5000/
```

---

## 🚀 Production Deployment Options

### **Option 1: Render.com (RECOMMENDED - Easiest)**

#### Step 1: Prepare Repository
1. Create a GitHub account if you don't have one
2. Create a new repository: `recommendation-engine`
3. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Milestone 4 - Web App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/recommendation-engine.git
   git push -u origin main
   ```

#### Step 2: Create Render.com Account
1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub account
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure deployment:
   - **Name:** `recommendation-engine`
   - **Environment:** Python 3
   - **Build Command:** `bash render-build.sh`
   - **Start Command:** `gunicorn src.app:app --bind 0.0.0.0:$PORT`
   - **Plan:** Free tier (auto-suspend, upgrade later if needed)

**IMPORTANT:** The build script `render-build.sh` automatically installs Git LFS and pulls the large CSV data files (316 MB) required for the recommendation engine.

#### Step 3: Environment Variables
Add in Render dashboard → Environment:
```
FLASK_ENV=production
FLASK_DEBUG=False
```

#### Step 4: Deploy
Click "Create Web Service" and wait for deployment (2-3 minutes)

**Your app will be live at:** `https://recommendation-engine.onrender.com`

---

### **Option 2: PythonAnywhere (Simple Alternative)**

#### Step 1: Create Account
1. Go to [https://www.pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for a free account
3. Go to "Web" → "Add a new web app"

#### Step 2: Configure Python
- Select Python 3.10
- Choose Flask framework
- Specify directory: `/home/your_username/recommendation-engine`

#### Step 3: Upload Code
1. Use the "Files" section to upload your code
2. Or use the bash terminal:
   ```bash
   cd /home/your_username
   git clone https://github.com/YOUR_USERNAME/recommendation-engine.git
   cd recommendation-engine
   mkvirtualenv --python=/usr/bin/python3.10 venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### Step 4: Edit WSGI File
In the "Web" tab, edit WSGI configuration:
```python
import sys
path = '/home/your_username/recommendation-engine'
if path not in sys.path:
    sys.path.append(path)

from src.app import app as application
```

#### Step 5: Reload and Deploy
Click "Reload" to start your web app

**Your app will be live at:** `https://your_username.pythonanywhere.com`

---

### **Option 3: AWS EC2 (Most Control, More Complex)**

#### Step 1: Create EC2 Instance
1. Go to AWS Console
2. Launch an Ubuntu 20.04 instance (t2.micro free tier eligible)
3. Configure security group to allow ports 80, 443, 5000

#### Step 2: Connect and Setup
```bash
# SSH into instance
ssh -i your_key.pem ubuntu@your_instance_ip

# Install Python and dependencies
sudo apt update
sudo apt install python3-pip python3-venv git -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/recommendation-engine.git
cd recommendation-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

#### Step 3: Create Systemd Service
Create `/etc/systemd/system/recommendation-api.service`:
```ini
[Unit]
Description=AI Recommendation Engine API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/recommendation-engine
Environment="PATH=/home/ubuntu/recommendation-engine/venv/bin"
ExecStart=/home/ubuntu/recommendation-engine/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 src.app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable recommendation-api
sudo systemctl start recommendation-api
```

#### Step 4: Setup Nginx Reverse Proxy
```bash
sudo apt install nginx -y
sudo systemctl start nginx
```

Configure `/etc/nginx/sites-available/default`:
```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    server_name your_domain_or_ip;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Restart Nginx:
```bash
sudo systemctl reload nginx
```

---

## 📊 Testing Deployed Application

After deployment, test your live application:

```bash
# Test health check
curl https://your-deployed-url/api/health

# Test sample users
curl https://your-deployed-url/api/users/sample

# Test recommendations (replace USER_ID)
curl -X POST https://your-deployed-url/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"ATVPDKIKX0DER","top_n":5}'
```

Or use the web interface:
```
https://your-deployed-url/
```

---

## 🔧 Troubleshooting

### Models Not Loading
- Check that `models/` directory contains all `.pkl` files
- Verify `data/processed/` directory has CSV files
- Check `config.yaml` paths are correct

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Static Files Not Loading
- Verify `static/` folder structure:
  ```
  static/
  ├── index.html
  ├── css/
  │   └── style.css
  └── js/
      └── app.js
  ```
- Clear browser cache (Ctrl+Shift+Delete)

### API Timeout Issues
- Increase timeout in test_webapp.py: `TIMEOUT = 30`
- May be normal on first request while models load
- Use larger server instance if persistent

### CORS Issues
- Flask-CORS should handle it, but if not:
  - Check CORS headers in response
  - Verify frontend URL matches deployment domain

---

## 📈 Scaling for Production

### For Render.com:
1. Upgrade from free plan to paid
2. Increase number of workers: `gunicorn -w 8 -b 0.0.0.0:$PORT src.app:app`
3. Add caching: Configure Redis addon

### For PythonAnywhere:
1. Upgrade to paid plan for more CPU/memory
2. Use multiple web app instances with load balancer

### For AWS:
1. Use larger instance type (t3.small)
2. Add auto-scaling group
3. Use AWS RDS for data (optional)
4. Add CloudFront CDN for static files

---

## 🔐 Security Best Practices

1. **Environment Variables:**
   - Never commit `.env` files
   - Use deployment platform's environment variables
   - Set `FLASK_DEBUG=False` in production

2. **HTTPS:**
   - Enable SSL/TLS on all platforms
   - Use free certificates (Let's Encrypt on AWS)

3. **Input Validation:**
   - Already implemented in API endpoints
   - Validate all user inputs server-side

4. **Rate Limiting:**
   - Consider adding flask-limiter for production
   - Implement request throttling

---

## 📝 Monitoring & Logging

### Render.com Logs:
- Dashboard → Logs tab
- View in real-time

### PythonAnywhere Logs:
- "Web" tab → Error log
- "Files" tab → error_log.txt

### AWS EC2:
```bash
# Check service status
sudo systemctl status recommendation-api

# View logs
journalctl -u recommendation-api -f
sudo tail -f /var/log/nginx/access.log
```

---

## 💡 Example Deployment Timeline

| Task | Time |  
|------|------|  
| Setup Render account | 5 min |  
| Configure deployment | 5 min |  
| Push code to GitHub | 2 min |  
| Deploy via Render | 3-5 min |  
| Test endpoints | 5 min |  
| **Total** | **~20-25 min** |  

---

## 🎉 You're Done!

Your Milestone 4 is complete:
- ✅ Backend API functional
- ✅ Frontend web interface built
- ✅ Application deployed to production
- ✅ All endpoints tested and working

**Share your live app URL:** `https://your-deployed-url`

---

## 📞 Support & Next Steps

For issues or questions:
1. Check Troubleshooting section above
2. Review API error messages
3. Check deployment platform logs
4. Verify all required files are included

For future improvements:
1. Add authentication/user accounts
2. Implement recommendation caching
3. Add A/B testing for models
4. Setup CI/CD pipeline with GitHub Actions
5. Add database (PostgreSQL) for persistence

