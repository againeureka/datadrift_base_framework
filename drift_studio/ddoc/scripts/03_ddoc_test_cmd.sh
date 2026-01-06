# 'my-first-exp'라는 이름으로, 학습률(lr)을 0.0001로 설정하여 실험 큐에 추가
$ ddoc exp run exp01 --params '{"train": {"lr": 0.0001}}'

# 변경 사항만 확인하는 Dry Run
$ ddoc exp run exp06 --params '{"train": {"epochs": 1}}'

# 'HEAD' 커밋(현재 작업 디렉토리)의 상태와 'exp-12345' 실험을 비교
$ ddoc exp show --name HEAD --baseline exp-12345 

# 'master' 브랜치의 베이스라인과 'my-first-exp'를 비교
$ ddoc exp show --name my-first-exp --baseline master