# 문제 원인 되는 빌드 잔재 제거
pip install --upgrade pip
rm -rf build/ dist/ *.egg-info
pip install --no-cache-dir -e . 
