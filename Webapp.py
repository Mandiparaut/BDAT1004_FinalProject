import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title("Stock Market Data Prediction App")

stocks = ("INTC", "NVDA", "AMD", "QCOM", "TSMC34.SA")
selected_stock = st.selectbox("Select Dataset for Prediction", stocks)

n_year = st.slider("Years of Prediction:", 1, 6)
period = n_year * 365

@st.cache_data
def load_data(ticker):
  data = yf.download(ticker, START, TODAY)
  data.reset_index(inplace=True)
  return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw Data')
st.write(data.tail())

#moving averages
ma250= data.Close.rolling(250).mean() #black
ma100= data.Close.rolling(100).mean() #red
ma50= data.Close.rolling(50).mean() #green

#Plotting separated new graphs for moving averages
#50 & 100
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open'))
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close'))

fig2.add_trace(go.Line(x=data['Date'], y=ma50, name='MA 50 days', line= dict(color='green')))
fig2.add_trace(go.Line(x=data['Date'], y=ma100, name='MA 100 days', line= dict(color='red')))

fig2.update_layout(title_text="Time Series Data for MA 50 days & 100 days", xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)

#100 & 250
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close'))

fig3.add_trace(go.Line(x=data['Date'], y=ma250, name='MA 250 days', line= dict(color='black')))
fig3.add_trace(go.Line(x=data['Date'], y=ma100, name='MA 100 days', line= dict(color='red')))

fig3.update_layout(title_text="Time Series Data for MA 100 days & 250 days", xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig3)


#MAIN PLOT
def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open'))
  fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close'))

  fig.add_trace(go.Line(x=data['Date'], y=ma250, name='MA 250 days', line= dict(color='black')))
  fig.add_trace(go.Line(x=data['Date'], y=ma100, name='MA 100 days', line= dict(color='red')))
  fig.add_trace(go.Line(x=data['Date'], y=ma50, name='MA 50 days', line= dict(color='green')))

  fig.update_layout(title_text="Time Series Data", xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)

plot_raw_data()


#forecasting with Prophet
import pandas as pd
df_train_for_prophet = data[['Date', 'Close']].copy()
df_train_for_prophet= df_train_for_prophet.rename(columns={"Date": "ds", "Close":"y"})

#Instantiate the Prophet model
model = Prophet() 
model.fit(df_train_for_prophet)
future = model.make_future_dataframe(periods=period) #Create a DataFrame for future dates
forecast = model.predict(future) #Forecast future data


#Plot the forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

#Plotting a forecast
st.write('Forecast Data')
figure1 = plot_plotly(model, forecast)
st.plotly_chart(figure1)

st.write('forecast components')
figure2 = model.plot_components(forecast)
st.write(figure2)
