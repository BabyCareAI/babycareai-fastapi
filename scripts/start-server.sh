#!/bin/bash

echo "--------------- release/37-v0.4.2 배포 시작(3) -----------------"

cd /home/ubuntu/babycareai-server || {
  echo "디렉토리 이동 실패: /home/ubuntu/babycareai-server"
  exit 1
}

# 기존 컨테이너 중지 및 제거
docker compose down

# 최신 이미지 가져오기
docker pull 293337163237.dkr.ecr.ap-northeast-2.amazonaws.com/babycareai/fastapi-server:latest

# 백그라운드로 전체 서비스 시작
docker compose up -d

# 필요 없는 fastapi-server 이미지 정리
docker images | grep "fastapi-server" | grep -v "latest" | awk '{print $3}' | xargs -r docker rmi -f

echo "--------------- 서버 배포 완료 -----------------"