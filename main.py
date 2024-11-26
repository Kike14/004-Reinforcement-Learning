from env import HybridTradingEnv
from agent import Agent
from data_utils import download_data, preprocess_data, split_and_filter_by_year
from scenario_generator import load_generator_model, generate_scenarios

# Configuración inicial
ticker = "MANU"  # Cambia al ticker que necesites
start_date = '2014-01-01'
end_date = '2023-12-31'

# Descargar y procesar datos reales
data = download_data(ticker, start_date, end_date)

# Columnas clave para análisis
important_columns = [
    'Close', 'Open', 'High', 'Low', 'volume_adi', 'volume_obv', 'volume_vwap',
    'volatility_bbh', 'volatility_bbl', 'volatility_atr', 'trend_macd',
    'trend_macd_signal', 'trend_sma_fast', 'trend_sma_slow', 'trend_adx',
    'momentum_rsi', 'momentum_stoch', 'momentum_wr'
]

# Preprocesar datos reales
prices, _, _, _, _ = preprocess_data(data)

# Dividir los datos reales en datasets por año
real_data_datasets = split_and_filter_by_year(data, important_columns)

# Cargar modelo generador y generar escenarios simulados
generator = load_generator_model("generador.keras")  # Ruta al modelo generador
simulated_datasets = generate_scenarios(generator, prices, num_scenarios=1000)

# Filtrar columnas clave en datasets simulados
simulated_datasets = [df[important_columns] for df in simulated_datasets]

# Crear entorno y agente
env = HybridTradingEnv(real_data=real_data_datasets, simulated_datasets=simulated_datasets)
agent = Agent(env)

# Entrenar al agente
agent.train()

# Guardar el modelo
agent.save_model("trading_agent_model.keras")