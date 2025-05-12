from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# mysql docker container
# docker run -e MYSQL_ROOT_PASSWORD=password123 -p 3306:3306 -d mysql
# database name: test-db

# pymysql
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:password123@localhost/test-db" #로컬 Docker DB URL
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://admin:test-babycareai-db@test-babycareai-db.cxg020umkcqs.ap-northeast-2.rds.amazonaws.com:3306/test-db" # AWS RDS URL

# aiomysql
SQLALCHEMY_DATABASE_URL = "mysql+aiomysql://root:password123@localhost/test-db"




# 동기 엔진 생성
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL
# )

# 비동기 엔진 생성
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,  # 쿼리 로그 출력 (디버깅 용도로 활용)
)

# 세션을 생성할 sessionmaker 정의
# SessionLocal = sessionmaker(
#     autocommit=False,
#     autoflush=False,
#     bind=engine,
# )

# 비동기 세션을 생성할 sessionmaker 정의
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()
# declarative_base는 모든 클래스의 부모 클래스이다.


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

async def get_db():
    async with SessionLocal() as session:
        yield session


