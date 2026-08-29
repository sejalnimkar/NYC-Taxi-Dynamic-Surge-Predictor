FROM apache/spark:4.0.2-scala2.13-java17-python3-r-ubuntu

USER root

RUN pip install --no-cache-dir \
    notebook \
    pandas \
    pyarrow \
    pyspark \
    requests \
    snowflake-connector-python \
    snowflake-snowpark-python \
    geopandas \
    folium \
    python-dotenv \
    dbt-core \
    dbt-snowflake \
    mlflow \
    lightgbm \
    xgboost \ 
    optuna \
    "optuna-integration[mlflow]" 


WORKDIR /home/jovyan/work