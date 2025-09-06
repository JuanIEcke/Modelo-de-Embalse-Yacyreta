import streamlit as st
import subprocess
import os
import shutil
import uuid
import time
import pandas as pd
import plotly.graph_objects as go

# --- 0. CONFIGURACIÓN INICIAL Y CSS PERSONALIZADO ---
custom_css = f"""
<style>
/* Clase para el título principal, ajustada sin imagen de fondo */
.stApp > header {{
    background-color: transparent;
}}

.st-emotion-cache-18ni7ap {{ /* Ajuste del margen superior para el contenido principal */
    margin-top: -1cm;
}}

h1.st-emotion-cache-10trblm {{ /* Selector del H1, puede variar levemente en versiones de Streamlit */
    text-align: center;
    color: #31333F; /* Color de texto oscuro, típico de Streamlit */
    text-shadow: none; /* Sin sombra */
    font-size: 2.5em; 
    padding-top: 0.5cm; 
}}

/* Color de fondo para el sidebar */
.st-emotion-cache-1cypcdb {{ /* Sidebar wrapper */
    background-color: rgba(240, 242, 246, 0.95); /* Fondo casi opaco para el sidebar */
    padding: 1rem;
    border-radius: 10px;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Modelo de la Central Hidroeléctrica YACYRETÁ",
    page_icon="🌊",
    layout="wide"
)

st.title("💧⚡ Modelo de la Central Hidroeléctrica YACYRETÁ ⚡💧")

st.markdown("Esta aplicación web permite configurar y ejecutar el modelo de simulación (`modelo_backend.py`).")
st.info("**Instrucciones:**\n1. Los datos se cargan automáticamente desde GitHub.\n2. Ajusta los parámetros en el panel de la izquierda.\n3. Haz clic en 'Ejecutar Simulación'.\n4. Espera a que la simulación termine y descarga los resultados.")

# --- 2. PANEL LATERAL DE CONFIGURACIÓN ---
st.sidebar.title("⚙️ Configuración de la Simulación")
st.sidebar.header("1. Archivos de Entrada y Salida")

st.sidebar.success("✅ Datos de entrada cargados desde GitHub.")
nombre_archivo_salida = st.sidebar.text_input(
    "Nombre del archivo de resultados:",
    value="ResultadosCorrida.xlsx"
)

# (El resto del sidebar no cambia)
with st.sidebar.expander("2. Parámetros Principales de Simulación", expanded=True):
    potencia_objetivo_mw = st.number_input("Potencia Objetivo [MW]", min_value=100.0, max_value=5000.0, value=3100.0, step=50.0)
    modo_verificacion = st.selectbox("Modo de Verificación", options=["Desactivado", "Activado"], index=0)
with st.sidebar.expander("3. Parámetros del Embalse", expanded=True):
    nivel_minimo_embalse_m = st.number_input("Nivel Mínimo del Embalse [m]", value=81.5)
    nivel_maximo_operativo_m = st.number_input("Nivel Máximo Operativo [m]", value=83.5)
    st.write("Nivel Inicial del Embalse [m]")
    use_avg_level = st.checkbox("Calcular como promedio de min/max", value=True)
    if use_avg_level:
        nivel_embalse_inicial = (nivel_minimo_embalse_m + nivel_maximo_operativo_m) / 2
        st.text(f"Nivel inicial calculado: {nivel_embalse_inicial:.2f} m")
    else:
        nivel_embalse_inicial = st.number_input("Nivel Inicial (manual) [m]", value=82.5, min_value=nivel_minimo_embalse_m, max_value=nivel_maximo_operativo_m)
with st.sidebar.expander("4. Configuración de Turbinas", expanded=False):
    st.subheader("Cantidad de Turbinas")
    cant_turbinas_viejas = st.number_input("Cantidad de turbinas 'viejas'", min_value=0, max_value=20, value=17, step=1)
    cant_turbinas_nuevas = st.number_input("Cantidad de turbinas 'nuevas'", min_value=0, max_value=5, value=3, step=1)
with st.sidebar.expander("5. Coeficientes Físicos y de Modelo", expanded=False):
    latitud_grados = st.number_input("Latitud Geográfica [grados decimales]", value=-27.57, format="%.2f")
    coeficiente_embalse = st.slider("Coeficiente de Embalse (para evaporación)", min_value=0.5, max_value=1.0, value=0.85, step=0.01)
    perdida_carga_m = st.number_input("Pérdida de Carga Promedio [m]", value=0.37, format="%.2f")
with st.sidebar.expander("6. Ajustes Avanzados de Convergencia", expanded=False):
    tolerancia_nivel = st.number_input("Tolerancia de Nivel para Convergencia [m]", value=1e-4, format="%.e")
    max_iteraciones = st.number_input("Máximo de Iteraciones de Continuidad", min_value=10, max_value=500, value=100, step=10)


# --- 3. LÓGICA DE EJECUCIÓN Y VISUALIZACIÓN ---
if st.sidebar.button("▶️ Ejecutar Simulación", type="primary"):
    
    # --- URL de tus datos en GitHub ---
    url_datos_github = "https://github.com/JuanIEcke/Modelo-de-Embalse-Yacyreta/raw/main/Datos.xlsx"

    # Verificamos que el backend exista
    if not os.path.exists("modelo_backend.py"):
        st.error("❌ No se encontró el archivo 'modelo_backend.py' en el repositorio.")
    else:
        run_dir = f"temp_run_{uuid.uuid4().hex}"
        os.makedirs(run_dir, exist_ok=True)
        
        try:
            # Intentamos descargar y guardar el archivo Excel desde GitHub
            df_github = pd.read_excel(url_datos_github, engine='openpyxl')
            datos_path = os.path.join(run_dir, "Datos.xlsx")
            # Guardamos una copia en el directorio temporal para que el backend lo pueda leer
            df_github.to_excel(datos_path, index=False)

        except Exception as e:
            st.error(f"❌ Error al cargar datos desde GitHub: {e}")
            st.warning("Verificá que la URL sea correcta y que el repositorio de GitHub sea público.")
            st.stop() # Detenemos la ejecución si no se pueden cargar los datos

        try:
            with open("modelo_backend.py", "r", encoding="utf-8") as f:
                script_content = f.read()

            # (Lógica de reemplazo de parámetros)
            script_content = script_content.replace(f"CANTIDAD_TURBINAS = {{\n    'vieja': 17,\n    'nueva': 3\n}}",f"CANTIDAD_TURBINAS = {{\n    'vieja': {cant_turbinas_viejas},\n    'nueva': {cant_turbinas_nuevas}\n}}")
            script_content = script_content.replace("POTENCIA_OBJETIVO_MW = 3100.0", f"POTENCIA_OBJETIVO_MW = {potencia_objetivo_mw}")
            script_content = script_content.replace("NIVEL_MINIMO_EMBALSE_M = 81.5", f"NIVEL_MINIMO_EMBALSE_M = {nivel_minimo_embalse_m}")
            script_content = script_content.replace("NIVEL_MAXIMO_OPERATIVO_M = 83.5", f"NIVEL_MAXIMO_OPERATIVO_M = {nivel_maximo_operativo_m}")
            script_content = script_content.replace("NIVEL_EMBALSE_INICIAL_PRIMERA_CORRIDA_M = (NIVEL_MINIMO_EMBALSE_M + NIVEL_MAXIMO_OPERATIVO_M) / 2", f"NIVEL_EMBALSE_INICIAL_PRIMERA_CORRIDA_M = {nivel_embalse_inicial}")
            script_content = script_content.replace('MODO_VERIFICACION = "Desactivado"', f'MODO_VERIFICACION = "{modo_verificacion}"')
            script_content = script_content.replace("LATITUD_YACYRETA_GRADOS = -27.57", f"LATITUD_YACYRETA_GRADOS = {latitud_grados}")
            script_content = script_content.replace("COEFICIENTE_EMBALSE = 0.85", f"COEFICIENTE_EMBALSE = {coeficiente_embalse}")
            script_content = script_content.replace("PERDIDA_CARGA_PROMEDIO_M = 0.37", f"PERDIDA_CARGA_PROMEDIO_M = {perdida_carga_m}")
            script_content = script_content.replace('NOMBRE_ARCHIVO_SALIDA = "ResultadosCorrida.xlsx"', f'NOMBRE_ARCHIVO_SALIDA = "{nombre_archivo_salida}"')
            script_content = script_content.replace("TOLERANCIA_NIVEL = 1e-4", f"TOLERANCIA_NIVEL = {tolerancia_nivel}")
            script_content = script_content.replace("MAX_ITERACIONES_CONTINUIDAD = 100", f"MAX_ITERACIONES_CONTINUIDAD = {max_iteraciones}")
            
            temp_script_path = os.path.join(run_dir, "temp_modelo_backend.py")
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            st.header("Salida de la Simulación en Tiempo Real")
            log_placeholder = st.empty()
            log_output = ""
            
            with st.spinner("La simulación se está ejecutando... por favor espera."):
                comando = ["python", "-X", "utf8", "temp_modelo_backend.py"]
                process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=run_dir, text=True, encoding="utf-8")
                for line in iter(process.stdout.readline, ''):
                    log_output += line
                    log_placeholder.code(log_output, language="log")
                process.stdout.close()
                return_code = process.wait()

            if return_code == 0:
                st.success("✅ Simulación completada con éxito.")
                
                resultado_path = os.path.join(run_dir, nombre_archivo_salida)
                if os.path.exists(resultado_path):
                    st.header("📊 Visualización de Resultados")
                    df_resultados = pd.read_excel(resultado_path, sheet_name='Resultados Diarios')
                    df_resultados['TIEMPO [dia]'] = pd.to_datetime(df_resultados['TIEMPO [dia]'])

                    tab1, tab2, tab3 = st.tabs(["💧 Niveles", "🌊 Caudales", "⚡ Potencia"])

                    with tab1:
                        # (La lógica de los gráficos no cambia)
                        st.subheader("Evolución del Nivel del Embalse")
                        fig_niveles = go.Figure()
                        fig_niveles.add_trace(go.Scatter(x=df_resultados['TIEMPO [dia]'], y=df_resultados['Nivel Final Ajustado [m]'], mode='lines', name='Nivel del Embalse', line=dict(color='royalblue')))
                        fig_niveles.add_hline(y=nivel_maximo_operativo_m, line_dash="dash", line_color="red", annotation_text=f"Nivel Máximo ({nivel_maximo_operativo_m} m)", annotation_position="bottom right")
                        fig_niveles.add_hline(y=nivel_minimo_embalse_m, line_dash="dash", line_color="orange", annotation_text=f"Nivel Mínimo ({nivel_minimo_embalse_m} m)", annotation_position="bottom right")
                        fig_niveles.update_layout(xaxis_title='Fecha', yaxis_title='Nivel [m]', template='plotly_white', height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                        st.plotly_chart(fig_niveles, use_container_width=True)

                    with tab2:
                        st.subheader("Caudal Entrante a la Represa")
                        fig_q_entrante = go.Figure()
                        fig_q_entrante.add_trace(go.Scatter(x=df_resultados['TIEMPO [dia]'], y=df_resultados['Q Entrante [m³/s]'], mode='lines', name='Caudal Entrante', line=dict(color='green')))
                        fig_q_entrante.update_layout(xaxis_title='Fecha', yaxis_title='Caudal [m³/s]', template='plotly_white', height=500)
                        st.plotly_chart(fig_q_entrante, use_container_width=True)

                        st.subheader("Caudales Salientes (Turbinado y Vertido)")
                        fig_q_saliente = go.Figure()
                        fig_q_saliente.add_trace(go.Scatter(x=df_resultados['TIEMPO [dia]'], y=df_resultados['Q Turbinado Real [m³/s]'], mode='lines', name='Caudal Turbinado', line=dict(color='blue')))
                        fig_q_saliente.add_trace(go.Scatter(x=df_resultados['TIEMPO [dia]'], y=df_resultados['Q Vertedero [m³/s]'], mode='lines', name='Caudal Vertido', line=dict(color='purple')))
                        fig_q_saliente.update_layout(xaxis_title='Fecha', yaxis_title='Caudal [m³/s]', template='plotly_white', height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                        st.plotly_chart(fig_q_saliente, use_container_width=True)

                    with tab3:
                        st.subheader("Potencia Generada Diariamente")
                        fig_pot_gen = go.Figure()
                        fig_pot_gen.add_trace(go.Scatter(x=df_resultados['TIEMPO [dia]'], y=df_resultados['Potencia Generada [MW]'], mode='lines', name='Potencia Generada', line=dict(color='darkorange')))
                        fig_pot_gen.update_layout(xaxis_title='Fecha', yaxis_title='Potencia [MW]', template='plotly_white', height=500)
                        st.plotly_chart(fig_pot_gen, use_container_width=True)

                        st.subheader("Curva de Duración de Potencia Ponderada")
                        df_resultados['Potencia Ponderada [MW]'] = (df_resultados['Potencia Generada [MW]'] * df_resultados['T Firme [dias]'] + df_resultados['Potencia Falla [MW]'] * df_resultados['T Falla [dias]'] + df_resultados['Potencia Secundaria [MW]'] * df_resultados['T Secundario [dias]'])
                        potencia_ordenada = df_resultados['Potencia Ponderada [MW]'].sort_values(ascending=False).reset_index(drop=True)
                        total_dias = len(potencia_ordenada)
                        porcentaje_tiempo = (potencia_ordenada.index + 1) / total_dias * 100
                        fig_pot_duracion = go.Figure()
                        fig_pot_duracion.add_trace(go.Scatter(x=porcentaje_tiempo, y=potencia_ordenada, mode='lines', name='Potencia Ponderada Ordenada', line=dict(color='firebrick')))
                        fig_pot_duracion.update_layout(xaxis_title='Porcentaje del Tiempo (%)', yaxis_title='Potencia Ponderada Diaria [MW]', template='plotly_white', height=500)
                        st.plotly_chart(fig_pot_duracion, use_container_width=True)

                    st.header("Descargar Resultados Completos")
                    with open(resultado_path, "rb") as f:
                        st.download_button(label=f"📥 Descargar {nombre_archivo_salida}", data=f, file_name=nombre_archivo_salida, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("⚠️ La simulación terminó, pero no se encontró el archivo de resultados.")
            else:
                st.error(f"❌ La simulación falló con el código de error: {return_code}")
        
        finally:
            time.sleep(1)
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
