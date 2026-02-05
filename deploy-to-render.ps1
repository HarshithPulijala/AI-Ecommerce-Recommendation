Write-Host "=== AI E-commerce Recommendation - Render Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Check git status
Write-Host "Checking repository status..." -ForegroundColor Yellow
cd "c:\Users\Anwith Pulijala\Desktop\Harshith.in\AI -ecommerce"
git status -sb

Write-Host "`n✓ Repository is synced with GitHub" -ForegroundColor Green
Write-Host ""

# Display deployment info
Write-Host "=== Deployment Configuration ===" -ForegroundColor Cyan
Write-Host "Repository: HarshithPulijala/AI-Ecommerce-Recommendation"
Write-Host "Branch: main"
Write-Host "Build Command: bash render-build.sh"
Write-Host "Start Command: gunicorn src.app:app --bind 0.0.0.0:`$PORT"
Write-Host ""

Write-Host "=== Opening Render Dashboard ===" -ForegroundColor Yellow
Write-Host "Please follow these steps in your browser:"
Write-Host ""
Write-Host "1. Login to Render (https://dashboard.render.com)" -ForegroundColor White
Write-Host "2. If you already have a web service:" -ForegroundColor White
Write-Host "   - Click on your service name" -ForegroundColor Gray
Write-Host "   - Click 'Manual Deploy' → 'Deploy latest commit'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. If this is your first deployment:" -ForegroundColor White
Write-Host "   - Click 'New +' → 'Web Service'" -ForegroundColor Gray
Write-Host "   - Select your GitHub repository" -ForegroundColor Gray
Write-Host "   - It will auto-detect render.yaml configuration" -ForegroundColor Gray
Write-Host "   - Click 'Create Web Service'" -ForegroundColor Gray
Write-Host ""

# Open Render dashboard
Write-Host "Opening Render Dashboard..." -ForegroundColor Green
Start-Process "https://dashboard.render.com"

Write-Host ""
Write-Host "=== What to Watch For ===" -ForegroundColor Cyan
Write-Host "Build logs should show:"
Write-Host "  ✓ Installing Git LFS" -ForegroundColor Gray
Write-Host "  ✓ Pulling LFS files (data: 316 MB, models: 714 MB)" -ForegroundColor Gray
Write-Host "  ✓ Installing Python dependencies" -ForegroundColor Gray
Write-Host "  ✓ Starting Gunicorn server" -ForegroundColor Gray
Write-Host ""
Write-Host "After deployment succeeds, test:" -ForegroundColor Yellow
Write-Host "  https://your-app.onrender.com/api/health"
Write-Host "  https://your-app.onrender.com/"
Write-Host ""
