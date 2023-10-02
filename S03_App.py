import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import joblib

modelo = joblib.load('logistic_regression_model.pkl')
colunas = pd.read_csv('Dados/dados_tratados.csv').columns[1:] # Removendo Churn

def TotalServiceTransformer(X):
    servicos = ['Multiplelines', 'Onlinesecurity', 'Onlinebackup',
                'Deviceprotection', 'Techsupport', 'Streamingtv',
                'Streamingmovies']
    contador_sim = lambda col: col.apply(lambda x: 1 if x == 'Yes' else 0)
    X['TotalServices'] = X[servicos].apply(contador_sim, axis=1).sum(axis=1)

def previsao_churn(dados_entrada):
    
    df_entrada = pd.DataFrame([dados_entrada], columns = colunas)
    TotalServiceTransformer(df_entrada)

    predicao = modelo.predict(df_entrada)
    probabilidade = modelo.predict_proba(df_entrada)[:, 1]

    return predicao[0], probabilidade[0]

# streamlit run S03_App.py

im = Image.open('Identidade Visual/icon.ico')

st.set_page_config(page_title = 'Churn Predictor Novexus', 
                   page_icon = im,
                   layout = 'wide')


image = Image.open('Identidade Visual/Logo (8).png')
st.image(image, use_column_width = True)


st.markdown("<h1 style='text-align: center;'>Churn Predictor</h1>", unsafe_allow_html=True)
st.write('Insira os detalhes do cliente abaixo para prever o desligamento da empresa.')

st.header('Informações gerais do cliente')

coluna1, coluna2, coluna3, coluna4 = st.columns(4)

with coluna1:
    genero = st.radio('Gênero', ['Female', 'Male'])

with coluna2:
    senior_citizen = st.radio('Cliente igual ou acima de 65 anos (0: Não, 1: Sim)', [0, 1])

with coluna3:
    parceiro = st.radio('Possui parceiro(a)', ['Yes', 'No'])

with coluna4:
    dependentes = st.radio('Possui dependentes', ['Yes', 'No'])

st.header('Serviços telefônicos')

coluna1, coluna2 = st.columns(2)

with coluna1:
    phone_service = st.radio('Assinatura de serviço telefônico', ['Yes', 'No'])

with coluna2:
    multiple_lines = st.radio('Assisnatura de mais de uma linha de telefone', ['Yes', 'No'])

st.header('Serviços de internet')

with st.expander("Opções de Assinatura"):
    internet_service = st.radio('Assinatura de um provedor internet', ['DSL', 'Fiber optic', 'No'])

with st.expander("Opções Adicionais"):
    online_security = st.radio('Assinatura adicional de segurança online', ['No', 'Yes'])
    online_backup = st.radio('Assinatura adicional de backup online', ['Yes', 'No'])
    device_protection = st.radio('Assinatura adicional de proteção no dispositivo', ['No', 'Yes'])
    tech_support = st.radio('Assinatura adicional de suporte técnico (menos tempo de espera)', ['Yes', 'No'])
    streaming_tv = st.radio('Assinatura de TV a cabo', ['Yes', 'No'])
    streaming_movies = st.radio('Assinatura de streaming de filmes', ['No', 'Yes'])

if internet_service == 'No':
    online_security = 'No'
    online_backup = 'No'
    device_protection = 'No'
    tech_support = 'No'
    streaming_tv = 'No'
    streaming_movies = 'No'

st.header('Assuntos financeiros do cliente')

tenure = st.slider('Meses de contrato', 0, 100, 1)

coluna1, coluna2, coluna3 = st.columns(3)

with coluna1:
    contrato = st.radio('Tipo de contrato', ['One year', 'Month-to-month', 'Two year'])

with coluna2:
    paperlessbilling = st.radio('O cliente prefere receber online a fatura', ['Yes', 'No'])

with coluna3:
    payment_method = st.radio('Forma de pagamento', ['Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'])


monthly_charges = st.number_input("Total de gastos por mês")
total_charges = st.number_input("Total de gastos")

dados_entrada = {
    'Gender': genero,
    'Seniorcitizen': senior_citizen,
    'Partner': parceiro,
    'Dependents': dependentes,
    'Tenure': tenure,
    'Phoneservice': phone_service,
    'Multiplelines': multiple_lines,
    'Internetservice': internet_service,
    'Onlinesecurity': online_security,
    'Onlinebackup': online_backup,
    'Deviceprotection': device_protection,
    'Techsupport': tech_support,
    'Streamingtv': streaming_tv,
    'Streamingmovies': streaming_movies,
    'Contract': contrato,
    'Paperlessbilling': paperlessbilling,
    'Paymentmethod': payment_method,
    'Charges_monthly': monthly_charges,
    'Charges_total': total_charges
}

predicao, probabilidade = previsao_churn(dados_entrada)
probabilidade_percent = round(probabilidade, 3) * 100

coluna1, coluna2 = st.columns([1, 2])

mensagem = f"<p style='font-size: 24px;; color: {'red' if probabilidade >= 0.5 else 'green'};'>"
mensagem += f"Este cliente <span style='font-weight: bold;'>{'está propenso' if probabilidade >= 0.5 else 'não está propenso'}"
mensagem += " a deixar a empresa</span></p>"

with coluna1:
    st.header('Probabilidade do cliente deixar a Novexus')
    st.markdown(mensagem, unsafe_allow_html=True)


with coluna2:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        number = {'suffix': "%", 'font': {'size': 50}},
        value = round(probabilidade, 3) * 100,
        gauge = {'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100], 
                        'ticktext': ['MÍNIMA', 'BAIXA', 'MÉDIA', 'ALTA', 'MÁXIMA'],
                        'tickwidth': 0.1, 'tickfont': {'size': 16, 'color': 'black'}},
                'bar': {'color': 'black'},
                'steps': [{'range': [0, 25], 'color': "#007A00"},
                        {'range': [25, 50], 'color': "#0063BF"}, 
                        {'range': [50, 75], 'color': "#FFCC00"},  
                        {'range': [75, 100], 'color': "#ED1C24"}]
                        }))

    st.plotly_chart(fig, use_container_width = True)









