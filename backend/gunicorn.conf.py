# 워커 수
import multiprocessing
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 1

# 워커로 uvicorn 사용
worker_class = "uvicorn.workers.UvicornWorker"

# 로그 
loglevel = "debug"
accesslog = "-"
errorlog = "-"

########################################## 수정 예정 ##########################################
# 바인드 주소 및 포트 
bind = "0.0.0.0:3000"

# 타임아웃 
# timeout = 60

# 배포 버전에서 삭제
# reload = True

# 파일에 로그 저장
# accesslog = "log/gunicorn/access.log"
# errorlog = "log/gunicorn/error.log"
########################################## 수정 예정 ##########################################
