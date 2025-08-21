# Databricks notebook source
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, ArrayType
from py4j.java_gateway import java_import
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName("dataPreparation").getOrCreate()

# COMMAND ----------

def plotGraph(df_spark, symbol): 
    # Filter and prepare dataframe
    df = df_spark.filter(F.col('symbol') == symbol).toPandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp'])
    df['ts_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['date'] = df['timestamp'].dt.date

    # --- Calculate SMA if not already in df ---
    if 'sma' not in df.columns:
        df['sma'] = df['close'].rolling(window=14).mean()

    # --- Calculate RSI if not already in df ---
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # --- Calculate EMA of ADR_Score if not already in df ---
    if 'ADR_Score' in df.columns and 'ema_ADR' not in df.columns:
        df['ema_ADR'] = df['ADR_Score'].ewm(span=14, adjust=False).mean()

    # --- Calculate color-coded volume ---
    df['volume_color'] = np.where(df['close'] >= df['open'], 
                                  'rgba(0,200,0,0.5)', 
                                  'rgba(200,0,0,0.5)')

    # --- Create subplots ---
    fig = sp.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.5, 0.25, 0.15, 0.1],
        subplot_titles=("Price & Volume", "MACD", "RSI", "ADR_Score"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}], [{}]]
    )

    # === PRICE CHART ===
    fig.add_trace(go.Candlestick(
        x=df['ts_str'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Candlestick', increasing_line_color='green',
        decreasing_line_color='red', showlegend=False
    ), row=1, col=1, secondary_y=False)

    # EMA
    fig.add_trace(go.Scatter(
        x=df['ts_str'], y=df['ema'],
        name='EMA', mode='lines',
        line=dict(color='orange', width=2, dash='dash')
    ), row=1, col=1, secondary_y=False)

    # SMA
    fig.add_trace(go.Scatter(
        x=df['ts_str'], y=df['sma'],
        name='SMA', mode='lines',
        line=dict(color='blue', width=2)
    ), row=1, col=1, secondary_y=False)

    # === VWAPSupport ===
    if 'VWAPSupport' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['ts_str'], y=df['VWAPSupport'],
            name="VWAPSupport",
            mode="lines",
            line=dict(color="black", width=2.5, dash="dot")
        ), row=1, col=1, secondary_y=False)

    # === VOLUME BARS ===
    fig.add_trace(go.Bar(
        x=df['ts_str'], y=df['volume'],
        name="Volume",
        marker_color=df['volume_color'],
        marker_line_width=0,
        opacity=0.3
    ), row=1, col=1, secondary_y=True)

    # === MACD CHART ===
    fig.add_trace(go.Bar(
        x=df['ts_str'], y=df['macdHist'], name='Histogram',
        marker_color=np.where(df['macdHist'] >= 0, 'rgba(0,200,0,0.8)', 'rgba(200,0,0,0.8)'),
        marker_line_width=0
    ), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['ts_str'], y=df['macd'],
        name='MACD', mode='lines',
        line=dict(color='purple', width=1.5)
    ), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df['ts_str'], y=df['signal'],
        name='Signal', mode='lines',
        line=dict(color='gold', width=1.5)
    ), row=2, col=1, secondary_y=False)

    # --- MACD histogram scale ---
    hist_max = max(abs(df['macdHist'].max()), abs(df['macdHist'].min())) * 2.0
    fig.update_yaxes(range=[-hist_max, hist_max], row=2, col=1, secondary_y=True)

    # --- MACD & Signal scale with padding ---
    macd_min = min(df['macd'].min(), df['signal'].min())
    macd_max = max(df['macd'].max(), df['signal'].max())
    macd_pad = (macd_max - macd_min) * 0.2
    fig.update_yaxes(range=[macd_min - macd_pad, macd_max + macd_pad], row=2, col=1, secondary_y=False)

    # === RSI CHART ===
    fig.add_trace(go.Scatter(
        x=df['ts_str'], y=df['rsi'],
        name='RSI', mode='lines',
        line=dict(color='teal', width=1.5)
    ), row=3, col=1)

    # RSI levels
    fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=3, col=1)

    # === EMA of ADR ===
    if 'ema_ADR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['ts_str'], y=df['ema_ADR'],
            name='EMA_ADR', mode='lines',
            line=dict(color='darkblue', width=2, dash='dot')
        ), row=4, col=1)

    # === SHADED BACKGROUNDS FOR DIFFERENT DAYS ===
    unique_dates = df['date'].unique()
    for i, d in enumerate(unique_dates):
        day_df = df[df['date'] == d]
        x_start = day_df['ts_str'].iloc[0]
        x_end = day_df['ts_str'].iloc[-1]
        fig.add_vrect(
            x0=x_start, x1=x_end,
            fillcolor="rgba(180, 180, 255, 0.3)" if i % 2 == 0 else "rgba(200, 200, 200, 0.3)",
            layer="below", line_width=0, row="all", col=1
        )

    # === LAYOUT ===
    fig.update_layout(
        title=f"{symbol} - Candlestick, Close, Volume, MACD, RSI, ADR_Score & EMA_ADR",
        height=1200,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        barmode='overlay',
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=df['ts_str'].tolist(),
            rangeslider=dict(visible=True, thickness=0.05),
            range=[len(df) - 50, len(df) - 1]
        ),
        xaxis2=dict(type='category', categoryorder='array', categoryarray=df['ts_str'].tolist()),
        xaxis3=dict(type='category', categoryorder='array', categoryarray=df['ts_str'].tolist()),
        xaxis4=dict(type='category', categoryorder='array', categoryarray=df['ts_str'].tolist())
    )

    # Remove left/right padding
    fig.update_xaxes(range=[df['ts_str'].iloc[0], df['ts_str'].iloc[-1]])

    # === ADD VERTICAL SPIKELINES ===
    fig.update_xaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=1)

    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="MACD", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Histogram", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="ADR / EMA_ADR", row=4, col=1)

    fig.show()


# COMMAND ----------

def readTableFromCSV(path):
    df = spark.read.csv(path,header=True,inferSchema=True,sep=',')
    return df

# COMMAND ----------

rawData = readTableFromCSV('/Volumes/workspace/default/data/merged_output_1755542386956.csv')\
    .withColumn('daydate',F.date_format(F.col('Timestamp'),"yyyy-MM-dd"))\
    .withColumn("timestamp", F.date_format(F.col("timestamp"),"yyyy-MM-dd HH:mm:ss"))\
    .withColumn('RSI',F.col('close')-F.col('open'))\
    .withColumn('symbol',F.concat_ws('|',F.col('symbol'),F.col('instrument_type')))\
    .filter(F.col('instrument_type')!='EXT')

# COMMAND ----------

extInfluencers = readTableFromCSV('/Volumes/workspace/default/data/merged_output_1755542386956.csv')\
    .withColumn('daydate',F.date_format(F.col('Timestamp'),"yyyy-MM-dd"))\
    .withColumn("timestamp", F.date_format(F.col("timestamp"),"yyyy-MM-dd HH:mm:ss"))\
    .withColumn('RSI',F.col('close')-F.col('open'))\
    .withColumn('symbol',F.concat_ws('|',F.col('symbol'),F.col('instrument_type')))\
    .filter(F.col('instrument_type')=='EXT')

# COMMAND ----------

volDataN50 = rawData.filter(F.col('instrument_type')=='N50_CONS').groupby('timestamp').agg(F.sum('volume').alias('volCon'))

# COMMAND ----------

newRawData = rawData\
    .filter((F.col('instrument_type')=='Index'))\
    .join(volDataN50,['timestamp'])\
    .withColumn('volume',F.col('volCon'))\
    .drop('volCon')

cols = newRawData.columns

newRawData = rawData\
        .filter(~(F.col('instrument_type')=='Index'))\
        .select(cols)\
        .union(newRawData)

# COMMAND ----------

hinkenAshiRawData_ = newRawData\
    .withColumn('close',(F.col('close')+F.col('open')+F.col('high')+F.col('low'))/F.lit(4))\
    .withColumn('openHA',F.lag(F.col('open')+F.col('close'),1).over(Window.partitionBy('symbol').orderBy(F.col("timestamp").asc()))/F.lit(2))\
    .withColumn('openHA',F.when(F.col('openHA').isNull(),F.col('open')).otherwise(F.col('openHA')))\
    .withColumn('highHA',F.greatest(F.col('high'),F.col('openHA'),F.col('close')))\
    .withColumn('lowHA',F.least(F.col('low'),F.col('openHA'),F.col('close')))\

vwapData = hinkenAshiRawData_.withColumn('PVsum',F.sum(F.col('close')*F.col('volume')).over(Window.partitionBy('symbol').orderBy(F.col('timestamp').asc()).rowsBetween(-4, 0)))\
    .withColumn('Psum',F.sum(F.col('volume')).over(Window.partitionBy('symbol').rowsBetween(-4, 0)))\
    .withColumn('VWAPSupport',F.col('PVsum')/F.when(F.col('Psum')==0,F.lit(1)).otherwise(F.col('Psum')))\
    .select('symbol','timestamp','VWAPSupport')

#HAIKEN ASHI CANDLES
hinkenAshiRawData = hinkenAshiRawData_\
    .join(vwapData,["symbol",'timestamp'])

# COMMAND ----------

@udf(StructType([
    StructField("list1", ArrayType(DoubleType())),
    StructField("list2", ArrayType(DoubleType()))
]))
def ema(lst,interval):
    emaList = []
    interval_slow = int(interval*2)
    k = 2 / (interval + 1)
    kSlow = 2 / (interval_slow + 1)
    if(len(lst)<=0):
        return [0]*len(lst),[0]*len(lst)
    else:
        emaSlow = []
        for i in range(len(lst)):
            if(i==0):
                emaList.append(lst[i])
                emaSlow.append(lst[i])
            else:
                emaList.append((lst[i]*k)+(emaList[i-1]*(1-k)))
                emaSlow.append((lst[i]*kSlow)+(emaSlow[i-1]*(1-kSlow)))
        return emaList,emaSlow

interval =14 

# COMMAND ----------

# DBTITLE 1,ADR
ADRData = hinkenAshiRawData_.withColumn(
        "OI_Volume_Score",
        F.log(1 + F.col("OpenInterest")) * F.log(1 + F.col("Volume"))
    )\
    .withColumn('lastPrice',F.lag(F.col('close'), 1).over(Window.partitionBy('symbol','instrument_type').orderBy(F.col('timestamp'))))\
    .withColumn('bulllingScore',F.when((F.col('instrument_type')=='PE')&(F.col('close')>F.col('lastPrice')),F.col('OI_Volume_Score'))\
    .when((F.col('instrument_type')=='CE')&(F.col('close')<F.col('lastPrice')),F.col('OI_Volume_Score')).otherwise(F.lit(0)))\
    .withColumn('bearishScore',F.when((F.col('instrument_type')=='PE')&(F.col('close')<F.col('lastPrice')),F.col('OI_Volume_Score'))\
    .when((F.col('instrument_type')=='CE')&(F.col('close')>F.col('lastPrice')),F.col('OI_Volume_Score')).otherwise(F.lit(0)))\
    .groupBy('timestamp')\
        .agg((F.sum('bulllingScore')/F.coalesce(F.when(F.sum("bearishScore") != 0, F.sum("bearishScore")),F.lit(1))).alias('ADR_Score'))\
    .withColumn('index',F.row_number().over(Window.orderBy(F.col("timestamp").asc())))\
    .withColumn('scores',F.collect_list(F.col('ADR_Score')).over(Window.partitionBy(F.lit(1)).orderBy(F.col("timestamp").asc())))\
    .withColumn('EMAData',ema(F.col('scores'),F.lit(interval)))\
    .withColumn('ema_ADR',F.expr("element_at(EMAData.list1,index)"))\
    .drop('EMAData')\
    .drop('scores','ADR_Score')

# COMMAND ----------

df_dict = hinkenAshiRawData\
    .withColumn('closePrices',F.collect_list(F.col('close')).over(Window.partitionBy('symbol').orderBy(F.col("timestamp").asc())))\
    .groupBy("symbol")\
    .agg(F.max("closePrices").alias("close_list"))
df_mid = hinkenAshiRawData.selectExpr('symbol','timestamp','close','openHA as open','highHA as high','lowHA as low','volume','OpenInterest').distinct()\
    .join(df_dict,['symbol'])\
    .withColumn('index',F.row_number().over(Window.partitionBy('symbol').orderBy(F.col("timestamp").asc())))\
    .drop('OpenInterest')

d_final = df_mid.withColumn('EMAData',ema(F.col('close_list'),F.lit(interval)))\
    .withColumn('ema',F.expr("element_at(EMAData.list1,index)"))\
    .withColumn('EMA_slow',F.expr("element_at(EMAData.list2,index)"))\
    .drop('EMAData')\
    .drop('close_list')\
    .withColumn('macd',F.col('EMA')-F.col('EMA_slow'))
macdData = d_final\
    .withColumn('macdLists',F.collect_list('macd').over(Window.partitionBy('symbol').orderBy(F.col("timestamp").asc())))\
    .groupBy("symbol")\
    .agg(F.max("macdLists").alias("macd_list"))

d_final = d_final\
    .join(macdData,['symbol'])\
    .withColumn('macdEMA',ema(F.col('macd_list'),F.lit(9)))\
    .withColumn('signal',F.expr("element_at(macdEMA.list1,index)"))\
    .drop('macdEMA')\
    .drop('macd_list')\
    .withColumn('macdHist',F.col('macd')-F.col('signal'))

# COMMAND ----------

nonOptionsData = hinkenAshiRawData.filter(~(F.col('instrument_type').isin('PE','CE')))
optionsData = hinkenAshiRawData.filter(F.col('instrument_type').isin('PE','CE'))

# COMMAND ----------

rolling_w = Window.partitionBy('symbol').orderBy(F.col("timestamp").asc()).rowsBetween(-1*interval, 0)
w = Window.partitionBy('symbol').orderBy(F.col("timestamp").asc())
df = hinkenAshiRawData\
    .withColumnRenamed('openHA','open')\
    .withColumnRenamed('highHA','high')\
    .withColumnRenamed('lowHA','low')\
    .withColumn("change", F.col("close") - F.lag("close", 1).over(w)) \
    .withColumn('profit',F.when(F.col('change')>0,F.col('change')).otherwise(F.lit(0)))\
    .withColumn('loss',F.when(F.col('change')<0,F.col('change')*F.lit(-1)).otherwise(F.lit(0)))\
    .withColumn('profit',F.avg('profit').over(rolling_w))\
    .withColumn('loss',F.sum('loss').over(rolling_w))\
    .withColumn('RSI',F.when((F.col('loss')==0)&(F.col('profit')>0),F.lit(100)).when((F.col('loss')==0)&(F.col('profit')==0),F.lit(0)).otherwise(F.lit(100)-(F.lit(100)/F.try_divide(F.lit(1)+F.col('profit'), F.col('loss')))))\
    .withColumn("SMA", F.avg("close").over(rolling_w))\
    .orderBy(F.col('symbol').asc(),F.col('timestamp').asc())\
    .select('symbol','timestamp','RSI','SMA','VWAPSupport')\
    .join(d_final,['symbol','timestamp'])\
    .join(ADRData,['timestamp'])


# COMMAND ----------

plotGraph(df,'NSE_FO|49472|24-07-2025|PE')
