import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
from flask import Flask, render_template_string

app = Flask(__name__)

# Função para obter dados históricos de uma criptomoeda
def get_crypto_data(crypto_id):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': '1', 'interval': 'hourly'}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    return pd.DataFrame(prices, columns=['timestamp', 'price'])

# Função para obter notícias de criptomoedas
def get_crypto_news(crypto_name):
    url = f"https://news.google.com/rss/search?q={crypto_name}+cryptocurrency"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    items = soup.findAll('item')
    news = []
    for item in items:
        news.append({
            'title': item.title.text,
            'link': item.link.text,
            'pubDate': item.pubDate.text
        })
    return news

# Lista de criptomoedas
cryptos = ['aave', 'polkadot', 'uniswap', 'solana', 'optimism', 'arbitrum', 'binancecoin', 'bitcoin', 'ethereum']

@app.route('/')
def index():
    # Coletar dados de todas as criptomoedas
    crypto_data = {crypto: get_crypto_data(crypto) for crypto in cryptos}

    # Processar dados para obter altas, baixas e médias
    crypto_summary = {}
    for crypto, df in crypto_data.items():
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        high = df['price'].max()
        low = df['price'].min()
        mean = df['price'].mean()
        crypto_summary[crypto] = {'high': high, 'low': low, 'mean': mean}

    # Aplicar regressão linear múltipla para previsões
    def prepare_data_for_regression(df):
        df['hour'] = df.index.hour
        X = df[['hour']]
        y = df['price']
        return X, y

    crypto_predictions = {}
    for crypto, df in crypto_data.items():
        X, y = prepare_data_for_regression(df)
        model = LinearRegression()
        model.fit(X, y)
        next_24_hours = np.array(range(24)).reshape(-1, 1)
        predictions = model.predict(next_24_hours)
        crypto_predictions[crypto] = predictions

    # Gerar template HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Monitor</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid black;
            }}
            th, td {{
                padding: 15px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Monitor de Criptomoedas</h1>
        <table>
            <tr>
                <th>Criptomoeda</th>
                <th>Alta (24h)</th>
                <th>Baixa (24h)</th>
                <th>Preço Médio (24h)</th>
                <th>Previsão para Próximas 24h</th>
            </tr>
            {rows}
        </table>
        <h2>Notícias Recentes</h2>
        {news}
    </body>
    </html>
    """

    # Preencher as linhas da tabela com os dados
    rows = ""
    for crypto, data in crypto_summary.items():
        prediction = crypto_predictions[crypto]
        prediction_str = ', '.join([f'{p:.2f}' for p in prediction])
        row = f"""
        <tr>
            <td>{crypto.capitalize()}</td>
            <td>{data['high']:.2f}</td>
            <td>{data['low']:.2f}</td>
            <td>{data['mean']:.2f}</td>
            <td>{prediction_str}</td>
        </tr>
        """
        rows += row

    # Obter notícias e formatar em HTML
    news_html = ""
    for crypto in cryptos:
        news = get_crypto_news(crypto)
        news_html += f"<h3>{crypto.capitalize()}</h3>"
        for item in news:
            news_html += f"<p><a href='{item['link']}'>{item['title']}</a> ({item['pubDate']})</p>"

    # Completar o template HTML
    html_output = html_template.format(rows=rows, news=news_html)

    return render_template_string(html_output)

if __name__ == '__main__':
    app.run(debug=True)
