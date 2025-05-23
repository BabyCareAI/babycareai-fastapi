#!/bin/bash

echo "--------------- 서버 배포 시작 -----------------"
docker stop fastapi-server || true
docker rm fastapi-server || true
docker pull 293337163237.dkr.ecr.ap-northeast-2.amazonaws.com/babycareai/fastapi-server:latest
docker run -d --name fastapi-server -p 8000:8000 293337163237.dkr.ecr.ap-northeast-2.amazonaws.com/babycareai/fastapi-server:latest
docker images | grep "fastapi-server" | grep -v "latest" | awk '{print $3}' | xargs -r docker rmi -f
echo "--------------- 서버 배포 완료 -----------------"