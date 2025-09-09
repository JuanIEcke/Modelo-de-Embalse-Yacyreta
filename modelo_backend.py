#Hola
import pandas as pd
import numpy as np
import datetime

# --- 0. PARÁMETROS DE OPERACIÓN Y CONSTANTES PRINCIPALES ---

# Parámetros de Turbinas
TURBINAS_DEFINICION = {
    'vieja': {
        'eficiencia': 0.88,
        'q_max_m3s': 830.0,
        'q_min_m3s': 376.0
    },
    'nueva': {
        'eficiencia': 0.94,
        'q_max_m3s': 830.0,
        'q_min_m3s': 376.0
    }
}
CANTIDAD_TURBINAS = {
    'vieja': 17,
    'nueva': 3
}
POTENCIA_OBJETIVO_MW = 3100.0

# Parámetros de Embalse
NIVEL_MINIMO_EMBALSE_M = 81.5
NIVEL_MAXIMO_OPERATIVO_M = 83.5
NIVEL_EMBALSE_INICIAL_PRIMERA_CORRIDA_M = (NIVEL_MINIMO_EMBALSE_M + NIVEL_MAXIMO_OPERATIVO_M) / 2
MODO_VERIFICACION = "Desactivado"

# Parámetros para el cálculo de evaporación (Hargreaves-Samani)
FECHA_INICIO_DATOS_REAL = '1994-09-01'
LATITUD_YACYRETA_GRADOS = -27.57
COEFICIENTE_EMBALSE = 0.85

# Parámetros de Pérdida de Carga
PERDIDA_CARGA_PROMEDIO_M = 0.37

# Rutas y Nombres de Hojas
RUTA_ARCHIVO_DATOS = "Datos.xlsx"
NOMBRE_HOJA_NIVEL_VOLUMEN = 'CurvaNivelVolumen'
NOMBRE_HOJA_NIVEL_SUPERFICIE = 'CurvaNivelSuperficie'
NOMBRE_HOJA_QT_NR = 'CurvaRestitucion'
NOMBRE_ARCHIVO_SALIDA = "ResultadosCorrida.xlsx"

# Constantes Físicas y de Conversión
DENSIDAD_AGUA = 1000
GRAVEDAD = 9.81
SEGUNDOS_EN_UN_DIA = 86400
CONVERSION_M3_A_HM3 = 1e6
FACTOR_DELTA_VOLUMEN = SEGUNDOS_EN_UN_DIA / CONVERSION_M3_A_HM3
CONSTANTE_QB = 1000 / 86400
GSC = 0.0820

# Parámetros de Tolerancia e Iteración
TOLERANCIA_QT = 0.01
MAX_ITERACIONES_QT = 100
TOLERANCIA_NIVEL = 1e-4
MAX_ITERACIONES_CONTINUIDAD = 100
TOLERANCIA_SUMA_DELTA_VOLUMEN = 1e-3

# Derivados de parámetros iniciales
POTENCIA_OBJETIVO_W = POTENCIA_OBJETIVO_MW * 1e6
Q_MAX_TOTAL_TURBINAS = sum(TURBINAS_DEFINICION[t]['q_max_m3s'] * CANTIDAD_TURBINAS.get(t, 0) for t in TURBINAS_DEFINICION)
Q_MIN_TOTAL_TURBINAS = sum(TURBINAS_DEFINICION[t]['q_min_m3s'] * CANTIDAD_TURBINAS.get(t, 0) for t in TURBINAS_DEFINICION)
LATITUD_YACYRETA_RADIANES = np.deg2rad(LATITUD_YACYRETA_GRADOS)

print(f"Capacidad máxima total de turbinado: {Q_MAX_TOTAL_TURBINAS} m³/s")
print(f"Capacidad mínima total de turbinado para operación: {Q_MIN_TOTAL_TURBINAS} m³/s")
print("---")

# --- 1. CARGA DE DATOS PRINCIPALES ---
try:
    df_bruto = pd.read_excel(RUTA_ARCHIVO_DATOS, header=None)
    fila_titulos = df_bruto[df_bruto.apply(lambda row: 'TIEMPO [dia]' in str(row.astype(str).tolist()), axis=1)].index.min()
    if pd.isna(fila_titulos):
        raise ValueError("No se encontraron los títulos de columna esperados.")
    df_original = pd.read_excel(RUTA_ARCHIVO_DATOS, skiprows=fila_titulos)
    df_original.columns = df_original.columns.str.strip()
    columnas_deseadas_principal = ['TIEMPO [dia]', 'Q Entrante [m³/s]', 'Precipitacion [mm]', 'Temperatura Minima [C]', 'Temperatura Maxima [C]']
    columnas_presentes = [col for col in columnas_deseadas_principal if col in df_original.columns]
    if len(columnas_presentes) != len(columnas_deseadas_principal):
        raise ValueError(f"Faltan columnas críticas en la hoja principal: {set(columnas_deseadas_principal) - set(columnas_presentes)}")
    df_original = df_original[columnas_presentes].copy()
    print("Datos de entrada extraídos correctamente:")
    print(df_original.head())
except (FileNotFoundError, ValueError) as e:
    print(f"Error en el procesamiento de datos: {e}")
    exit()
except Exception as e:
    print(f"Ocurrió un error inesperado durante la lectura: {e}")
    exit()
print("---")

# --- 2. CÁLCULO DE EVAPORACIÓN (Hargreaves-Samani) ---
def calcular_radiacion_extraterrestre(doy, latitud_rad):
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(latitud_rad) * np.tan(delta), -1.0, 1.0))
    Ra_mj_m2_dia = (24 * 60 / np.pi) * GSC * dr * (ws * np.sin(latitud_rad) * np.sin(delta) + np.cos(latitud_rad) * np.cos(delta) * np.sin(ws))
    return Ra_mj_m2_dia * 0.408

def calcular_evaporacion_hargreaves_samani(t_max, t_min, ra_mm_dia):
    t_avg = (t_max + t_min) / 2
    temp_diff_term = np.sqrt(np.maximum(0, t_max - t_min))
    et0 = 0.0023 * ra_mm_dia * temp_diff_term * (t_avg + 17.8)
    return np.maximum(0, et0)

print("Calculando evaporación diaria con Hargreaves-Samani...")
df_original['Fecha'] = pd.to_datetime(df_original['TIEMPO [dia]'])
df_original['DOY'] = df_original['Fecha'].dt.dayofyear
df_original['Ra [mm/dia]'] = df_original['DOY'].apply(lambda doy: calcular_radiacion_extraterrestre(doy, LATITUD_YACYRETA_RADIANES))
df_original['Evaporacion_HS_ET0 [mm]'] = calcular_evaporacion_hargreaves_samani(df_original['Temperatura Maxima [C]'], df_original['Temperatura Minima [C]'], df_original['Ra [mm/dia]'])
df_original['Evaporacion [mm]'] = df_original['Evaporacion_HS_ET0 [mm]'] * COEFICIENTE_EMBALSE
print("Cálculo de evaporación completado.")
print("---")

# --- 3. CARGA DE TABLAS DE INTERPOLACIÓN ---
try:
    # Corrección: Leer la hoja completa y luego limpiar las columnas
    curva_nv = pd.read_excel(RUTA_ARCHIVO_DATOS, sheet_name=NOMBRE_HOJA_NIVEL_VOLUMEN)
    curva_nv.columns = curva_nv.columns.str.strip()
    expected_cols_nv = ['Nivel [m]', 'Volumen [hm³]']
    curva_nv = curva_nv[expected_cols_nv].dropna().sort_values(by='Nivel [m]')
    
    curva_ns = pd.read_excel(RUTA_ARCHIVO_DATOS, sheet_name=NOMBRE_HOJA_NIVEL_SUPERFICIE)
    curva_ns.columns = curva_ns.columns.str.strip()
    expected_cols_ns = ['Nivel [m]', 'Superficie [km²]']
    curva_ns = curva_ns[expected_cols_ns].dropna().sort_values(by='Nivel [m]')
    
    curva_qt_nr = pd.read_excel(RUTA_ARCHIVO_DATOS, sheet_name=NOMBRE_HOJA_QT_NR)
    curva_qt_nr.columns = curva_qt_nr.columns.str.strip()
    expected_cols_qt_nr = ['Q Brazo Principal [m³/s]', 'Nivel Restitucion [m]']
    curva_qt_nr = curva_qt_nr[expected_cols_qt_nr].dropna().sort_values(by='Q Brazo Principal [m³/s]')
    
    print("Tablas de interpolación cargadas.")
except Exception as e:
    print(f"Error al cargar las tablas de interpolación: {e}")
    exit()
print("---")

# --- FUNCIONES DE INTERPOLACIÓN Y CÁLCULO ---
def volumen_por_nivel(nivel):
    return np.interp(nivel, curva_nv['Nivel [m]'], curva_nv['Volumen [hm³]'], left=curva_nv['Volumen [hm³]'].min(), right=curva_nv['Volumen [hm³]'].max())
def area_por_nivel(nivel):
    return np.interp(nivel, curva_ns['Nivel [m]'], curva_ns['Superficie [km²]'], left=curva_ns['Superficie [km²]'].min(), right=curva_ns['Superficie [km²]'].max())
def nr_por_qt(qt):
    return np.interp(qt, curva_qt_nr['Q Brazo Principal [m³/s]'], curva_qt_nr['Nivel Restitucion [m]'], left=curva_qt_nr['Nivel Restitucion [m]'].min(), right=curva_qt_nr['Nivel Restitucion [m]'].max())
def qt_por_potencia_y_salto_individual(potencia_objetivo_w, eficiencia, salto_m):
    return potencia_objetivo_w / (eficiencia * DENSIDAD_AGUA * GRAVEDAD * salto_m) if salto_m > 0 and eficiencia > 0 else 0.0
def nivel_por_volumen(volumen):
    return np.interp(volumen, curva_nv['Volumen [hm³]'], curva_nv['Nivel [m]'], left=curva_nv['Nivel [m]'].min(), right=curva_nv['Nivel [m]'].max())

def run_single_simulation(initial_embalse_level, is_verification_mode, caudal_modulo_serie=None):
    df = df_original.copy()
    
    # Initialize columns with appropriate data types
    numeric_cols = [
        'Nivel Embalse [m]', 'Volumen Embalse [hm³]', 'Area [km²]', 'Q Balance [m³/s]', 'Nr Calculado [m]',
        'Salto Calculado [m]', 'Qt Potencia Obj. [m³/s]', 'Q Turbinado Real [m³/s]', 'Q Vertedero [m³/s]',
        'Q Saliente Total [m³/s]', 'Nivel Final Ajustado [m]', 'Volumen Final Ajustado [hm³]',
        'Delta Volumen [hm³]', 'T Firme [dias]', 'T Falla [dias]', 'T Secundario [dias]',
        'Potencia Generada [MW]', 'Potencia Falla [MW]', 'Potencia Secundaria [MW]',
        'E Generada [MWh]', 'E Falla [MWh]', 'E Secundaria [MWh]', 'E Generada en Falla [MWh]',
        'E Generada en Secundario [MWh]'
    ]
    for col in numeric_cols:
        df[col] = np.nan
        
    df['Estado Nivel'] = ''
    
    nivel_embalse_inicial_ajustado = max(curva_nv['Nivel [m]'].min(), min(initial_embalse_level, curva_nv['Nivel [m]'].max()))
    df.at[0, 'Nivel Embalse [m]'] = nivel_embalse_inicial_ajustado
    df.at[0, 'Volumen Embalse [hm³]'] = volumen_por_nivel(nivel_embalse_inicial_ajustado)
    nivel_inicial_corrida_0 = df.at[0, 'Nivel Embalse [m]']
    EFICIENCIA_PROMEDIO = sum(TURBINAS_DEFINICION[t]['eficiencia'] * CANTIDAD_TURBINAS.get(t, 0) * TURBINAS_DEFINICION[t]['q_max_m3s'] for t in TURBINAS_DEFINICION) / Q_MAX_TOTAL_TURBINAS if Q_MAX_TOTAL_TURBINAS > 0 else 0.88

    for i in range(len(df)):
        ne_actual = df.at[i-1, 'Nivel Final Ajustado [m]'] if i > 0 else df.at[i, 'Nivel Embalse [m]']
        if i > 0:
            df.at[i, 'Nivel Embalse [m]'] = ne_actual
            df.at[i, 'Volumen Embalse [hm³]'] = df.at[i-1, 'Volumen Final Ajustado [hm³]']
        df.at[i, 'Area [km²]'] = area_por_nivel(ne_actual)
        qe_dia = df.at[i, 'Q Entrante [m³/s]']

        if is_verification_mode:
            qb_dia = 0.0
            q_turbinado_real_dia = caudal_modulo_serie
            df.at[i, 'Q Balance [m³/s]'] = qb_dia
            df.loc[i, 'Estado Nivel'] = 'Verificación'
            df.at[i, 'T Firme [dias]'] = 1.0
        else:
            precipitacion_dia = df.at[i, 'Precipitacion [mm]']
            evaporacion_dia = df.at[i, 'Evaporacion [mm]']
            area_dia = df.at[i, 'Area [km²]']
            qb_dia = (precipitacion_dia - evaporacion_dia) * area_dia * CONSTANTE_QB
            df.at[i, 'Q Balance [m³/s]'] = qb_dia

            qt_supuesto = Q_MAX_TOTAL_TURBINAS
            for _ in range(MAX_ITERACIONES_QT):
                nr = nr_por_qt(qt_supuesto)
                salto = ne_actual - nr - PERDIDA_CARGA_PROMEDIO_M
                if salto <= 0:
                    qt_nuevo = 0.0
                else:
                    qt_nuevo = qt_por_potencia_y_salto_individual(POTENCIA_OBJETIVO_W, EFICIENCIA_PROMEDIO, salto)
                    qt_nuevo = np.clip(qt_nuevo, Q_MIN_TOTAL_TURBINAS if qt_nuevo > 0 else 0, Q_MAX_TOTAL_TURBINAS)
                if abs(qt_nuevo - qt_supuesto) < TOLERANCIA_QT:
                    break
                qt_supuesto = qt_nuevo
            
            final_nr_iter = nr_por_qt(qt_supuesto)
            final_salto_iter = ne_actual - final_nr_iter - PERDIDA_CARGA_PROMEDIO_M
            df.at[i, 'Nr Calculado [m]'] = final_nr_iter
            df.at[i, 'Salto Calculado [m]'] = final_salto_iter
            df.at[i, 'Qt Potencia Obj. [m³/s]'] = qt_supuesto

            q_turbinado_real_dia = 0.0
            if ne_actual < NIVEL_MINIMO_EMBALSE_M:
                q_turbinado_real_dia = qe_dia + qb_dia
                q_turbinado_real_dia = np.clip(q_turbinado_real_dia, Q_MIN_TOTAL_TURBINAS if q_turbinado_real_dia > 0 else 0, Q_MAX_TOTAL_TURBINAS)
            elif ne_actual > NIVEL_MAXIMO_OPERATIVO_M:
                salto_secundario = ne_actual - nr_por_qt(Q_MAX_TOTAL_TURBINAS) - PERDIDA_CARGA_PROMEDIO_M
                if salto_secundario > 0:
                    qt_para_secundario = qt_por_potencia_y_salto_individual(POTENCIA_OBJETIVO_W, EFICIENCIA_PROMEDIO, salto_secundario)
                    q_turbinado_real_dia = np.clip(qt_para_secundario, Q_MIN_TOTAL_TURBINAS if qt_para_secundario > 0 else 0, Q_MAX_TOTAL_TURBINAS)
            else:
                q_turbinado_real_dia = qt_supuesto
            q_turbinado_real_dia = max(0.0, q_turbinado_real_dia)

        delta_volumen_preliminar_hm3 = (qe_dia + qb_dia - q_turbinado_real_dia) * FACTOR_DELTA_VOLUMEN
        volumen_inicio_dia = df.at[i, 'Volumen Embalse [hm³]']
        volumen_final_preliminar = volumen_inicio_dia + delta_volumen_preliminar_hm3
        nivel_final_preliminar = nivel_por_volumen(volumen_final_preliminar)

        nivel_final_ajustado = nivel_final_preliminar
        q_vertedero_dia = 0.0
        if nivel_final_preliminar > NIVEL_MAXIMO_OPERATIVO_M:
            nivel_final_ajustado = NIVEL_MAXIMO_OPERATIVO_M
            volumen_final_ajustado = volumen_por_nivel(NIVEL_MAXIMO_OPERATIVO_M)
            q_vertedero_dia = max(0.0, (volumen_final_preliminar - volumen_final_ajustado) / FACTOR_DELTA_VOLUMEN)
        elif nivel_final_preliminar < NIVEL_MINIMO_EMBALSE_M:
            nivel_final_ajustado = NIVEL_MINIMO_EMBALSE_M
            volumen_final_ajustado = volumen_por_nivel(NIVEL_MINIMO_EMBALSE_M)
        else:
            volumen_final_ajustado = volumen_final_preliminar

        df.at[i, 'Q Turbinado Real [m³/s]'] = q_turbinado_real_dia
        df.at[i, 'Q Vertedero [m³/s]'] = q_vertedero_dia
        df.at[i, 'Q Saliente Total [m³/s]'] = q_turbinado_real_dia + q_vertedero_dia
        df.at[i, 'Nivel Final Ajustado [m]'] = nivel_final_ajustado
        df.at[i, 'Volumen Final Ajustado [hm³]'] = volumen_final_ajustado
        df.at[i, 'Delta Volumen [hm³]'] = volumen_final_ajustado - volumen_inicio_dia

        if not is_verification_mode:
            ne_previo = df.at[i-1, 'Nivel Final Ajustado [m]'] if i > 0 else initial_embalse_level
            if ne_previo < NIVEL_MINIMO_EMBALSE_M and nivel_final_preliminar < NIVEL_MINIMO_EMBALSE_M:
                t_falla, t_normal, t_secundario, estado = 1.0, 0.0, 0.0, 'Falla'
            elif ne_previo > NIVEL_MAXIMO_OPERATIVO_M and nivel_final_preliminar > NIVEL_MAXIMO_OPERATIVO_M:
                t_falla, t_normal, t_secundario, estado = 0.0, 0.0, 1.0, 'Secundario'
            else:
                t_falla, t_normal, t_secundario, estado = 0.0, 1.0, 0.0, 'Normal'
                if nivel_final_preliminar < NIVEL_MINIMO_EMBALSE_M:
                    denominador = (ne_previo - nivel_final_preliminar)
                    if abs(denominador) > 1e-9:
                        t_falla = (NIVEL_MINIMO_EMBALSE_M - nivel_final_preliminar) / denominador
                        t_normal = 1.0 - t_falla
                    else:
                        t_falla, t_normal = 1.0, 0.0
                    estado = 'Falla'
                elif nivel_final_preliminar > NIVEL_MAXIMO_OPERATIVO_M:
                    denominador = (nivel_final_preliminar - ne_previo)
                    if abs(denominador) > 1e-9:
                        t_secundario = (nivel_final_preliminar - NIVEL_MAXIMO_OPERATIVO_M) / denominador
                        t_normal = 1.0 - t_secundario
                    else:
                        t_secundario, t_normal = 1.0, 0.0
                    estado = 'Secundario'
                t_falla, t_normal, t_secundario = [max(0, min(1.0, t)) for t in (t_falla, t_normal, t_secundario)]
                if abs(t_falla + t_normal + t_secundario - 1.0) > 1e-6:
                    t_falla, t_normal, t_secundario, estado = (1.0, 0.0, 0.0, 'Falla') if nivel_final_ajustado < NIVEL_MINIMO_EMBALSE_M else ((0.0, 0.0, 1.0, 'Secundario') if nivel_final_ajustado > NIVEL_MAXIMO_OPERATIVO_M else (0.0, 1.0, 0.0, 'Normal'))
            
            # Use .loc for multi-column assignment
            df.loc[i, ['T Firme [dias]', 'T Falla [dias]', 'T Secundario [dias]', 'Estado Nivel']] = [t_normal, t_falla, t_secundario, estado]
            
            salto = df.at[i, 'Salto Calculado [m]']
            potencia_generada_dia = EFICIENCIA_PROMEDIO * DENSIDAD_AGUA * GRAVEDAD * df.at[i, 'Q Turbinado Real [m³/s]'] * salto / 1e6 if salto > 0 else 0.0
            df.at[i, 'Potencia Generada [MW]'] = potencia_generada_dia

            # Use .loc for multi-column assignment
            df.loc[i, ['E Generada [MWh]', 'E Generada en Falla [MWh]', 'E Generada en Secundario [MWh]']] = [potencia_generada_dia * t * 24 for t in (t_normal, t_falla, t_secundario)]
            
            # Recalcular la potencia teórica en Falla y la energía de déficit
            q_falla_base = qe_dia + qb_dia
            salto_falla_calc = ne_actual - nr_por_qt(q_falla_base) - PERDIDA_CARGA_PROMEDIO_M
            potencia_falla_calculada = EFICIENCIA_PROMEDIO * DENSIDAD_AGUA * GRAVEDAD * q_falla_base * salto_falla_calc / 1e6 if salto_falla_calc > 0 else 0.0
            df.at[i, 'Potencia Falla [MW]'] = potencia_falla_calculada
            df.at[i, 'E Falla [MWh]'] = max(0, POTENCIA_OBJETIVO_MW - potencia_falla_calculada) * t_falla * 24

            # Recalcular la potencia teórica en Secundario y la energía desaprovechada
            potencia_secundaria_calculada = EFICIENCIA_PROMEDIO * DENSIDAD_AGUA * GRAVEDAD * df.at[i, 'Q Vertedero [m³/s]'] * salto / 1e6 if salto > 0 else 0.0
            df.at[i, 'Potencia Secundaria [MW]'] = potencia_secundaria_calculada
            df.at[i, 'E Secundaria [MWh]'] = potencia_secundaria_calculada * t_secundario * 24
            
    nivel_final_ultima_corrida = df.at[len(df)-1, 'Nivel Final Ajustado [m]']
    return df, nivel_inicial_corrida_0, nivel_final_ultima_corrida

# --- BUCLE DE ITERACIÓN PARA GARANTIZAR LA CONTINUIDAD ---
current_initial_level = NIVEL_EMBALSE_INICIAL_PRIMERA_CORRIDA_M
iteration = 0
converged = False
is_verification_mode_active = (MODO_VERIFICACION.lower() == "activado")
caudal_modulo_calc = df_original['Q Entrante [m³/s]'].mean() if is_verification_mode_active else None

print(f"\n--- Iniciando Proceso de Convergencia ({'Modo Verificación' if is_verification_mode_active else 'Modo Normal'}) ---")
if is_verification_mode_active:
    print(f"Caudal Módulo calculado: {caudal_modulo_calc:.2f} m³/s")

while not converged and iteration < MAX_ITERACIONES_CONTINUIDAD:
    iteration += 1
    df_current_run, initial_level_run, final_level_run = run_single_simulation(current_initial_level, is_verification_mode_active, caudal_modulo_calc)
    diff = abs(initial_level_run - final_level_run)
    print(f"\nCorrida #{iteration}: Nivel Inicial: {initial_level_run:.4f} m, Final: {final_level_run:.4f} m (Diferencia: {diff:.6f} m)")
    if diff < TOLERANCIA_NIVEL:
        converged = True
        print(f"¡Convergencia alcanzada en {iteration} iteraciones!")
    else:
        current_initial_level = final_level_run

final_df = df_current_run
final_initial_level = initial_level_run
final_last_level = final_level_run
if not converged:
    print(f"\nAdvertencia: La simulación no convergió después de {MAX_ITERACIONES_CONTINUIDAD} iteraciones.")
print("\n--- Simulación Finalizada ---")

# --- ANÁLISIS Y EXPORTACIÓN DE RESULTADOS ---
print("\nDataFrame con los resultados de la simulación (primeras y últimas filas):")
print(final_df.head())
print(final_df.tail())

if not is_verification_mode_active:
    energia_generada_normal_total = final_df['E Generada [MWh]'].sum()
    energia_generada_falla_total = final_df['E Generada en Falla [MWh]'].sum()
    energia_generada_secundaria_total = final_df['E Generada en Secundario [MWh]'].sum()
    energia_total_generada_real = energia_generada_normal_total + energia_generada_falla_total + energia_generada_secundaria_total
    energia_falla_deficit_total = final_df['E Falla [MWh]'].sum()
    energia_secundaria_desaprovechada_total = final_df['E Secundaria [MWh]'].sum()
    print(f"\n--- Resumen de Energía ---")
    print(f"🔋 Energía Total REAL Generada: {energia_total_generada_real:.2f} MWh")
    print(f"✅ Energía en Estado Normal: {energia_generada_normal_total:.2f} MWh")
    print(f"⚠️ Energía Generada en Falla: {energia_generada_falla_total:.2f} MWh")
    print(f"⚠️ Energía de DÉFICIT en Falla: {energia_falla_deficit_total:.2f} MWh")
    print(f"📘 Energía Generada en Secundario: {energia_generada_secundaria_total:.2f} MWh")
    print(f"📘 Energía Secundaria DESAPROVECHADA: {energia_secundaria_desaprovechada_total:.2f} MWh")

total_dias_simulacion = len(final_df)
tiempo_falla_dias = final_df['T Falla [dias]'].sum()
tiempo_normal_dias = final_df['T Firme [dias]'].sum()
tiempo_secundario_dias = final_df['T Secundario [dias]'].sum()
tiempo_garantia_dias = tiempo_normal_dias + tiempo_secundario_dias
porcentaje_falla = (tiempo_falla_dias / total_dias_simulacion) * 100
porcentaje_normal = (tiempo_normal_dias / total_dias_simulacion) * 100
porcentaje_secundario = (tiempo_secundario_dias / total_dias_simulacion) * 100
porcentaje_garantia = (tiempo_garantia_dias / total_dias_simulacion) * 100

print(f"\n📊 Resumen de Tiempos Operativos ({total_dias_simulacion} días) ---")
print(f"    📉 Tiempo en Falla: {tiempo_falla_dias:.2f} días ({porcentaje_falla:.2f}%)")
print(f"    ✅ Tiempo en Normal: {tiempo_normal_dias:.2f} días ({porcentaje_normal:.2f}%)")
print(f"    📘 Tiempo en Secundario: {tiempo_secundario_dias:.2f} días ({porcentaje_secundario:.2f}%)")
print(f"    Total de días calculados: {tiempo_falla_dias + tiempo_normal_dias + tiempo_secundario_dias:.2f} días")
print(f"    Tiempo en Garantía: {tiempo_garantia_dias:.2f} días ({porcentaje_garantia:.2f}%)")

diff_final_run = abs(final_initial_level - final_last_level)
verifica_continuidad_final = "Sí" if diff_final_run < TOLERANCIA_NIVEL else "No"
print(f"\n--- Verificación de Continuidad (Corrida Final) ---")
print(f"Diferencia: {diff_final_run:.6f} m. Verifica Continuidad: {verifica_continuidad_final}")

if is_verification_mode_active:
    sum_delta_volumen_hm3 = final_df['Delta Volumen [hm³]'].sum()
    verifica_sum_delta_volumen = "Sí" if abs(sum_delta_volumen_hm3) < TOLERANCIA_SUMA_DELTA_VOLUMEN else "No"
    print(f"\n--- Verificación de Suma de Delta Volumen (Modo Verificación) ---")
    print(f"Suma Total de Delta Volumen: {sum_delta_volumen_hm3:.6f} hm³. Verifica: {verifica_sum_delta_volumen}")

resumen_data = {
    'Métrica': ['Días Simulados', 'Nivel Inicial Embalse [m]', 'Nivel Final Embalse [m]', 'Diferencia Nivel [m]', 'Verifica Continuidad', 'Número de Iteraciones'],
    'Valor': [total_dias_simulacion, final_initial_level, final_last_level, diff_final_run, verifica_continuidad_final, iteration]
}
if is_verification_mode_active:
    resumen_data['Métrica'].extend(['Suma Total Delta Volumen [hm³]', 'Verifica Suma Delta Volumen'])
    resumen_data['Valor'].extend([sum_delta_volumen_hm3, verifica_sum_delta_volumen])
else:
    resumen_data['Métrica'].extend(['Tiempo en Falla [dias]', 'Tiempo en Falla [%]', 'Tiempo en Normal [dias]', 'Tiempo en Normal [%]',
                                    'Tiempo en Secundario [dias]', 'Tiempo en Secundario [%]', 'Tiempo en Garantía [dias]',
                                    'Tiempo en Garantía [%]', 'Energía Total REAL Generada [MWh]', 'Energía Generada en Normal [MWh]',
                                    'Energía Generada en Falla [MWh]', 'Energía de DÉFICIT en Falla [MWh]',
                                    'Energía Generada en Secundario [MWh]', 'Energía Secundaria DESAPROVECHADA [MWh]'])
    resumen_data['Valor'].extend([tiempo_falla_dias, porcentaje_falla, tiempo_normal_dias, porcentaje_normal, tiempo_secundario_dias,
                                  porcentaje_secundario, tiempo_garantia_dias, porcentaje_garantia, energia_total_generada_real,
                                  energia_generada_normal_total, energia_generada_falla_total, energia_falla_deficit_total,
                                  energia_generada_secundaria_total, energia_secundaria_desaprovechada_total])
df_resumen = pd.DataFrame(resumen_data)

try:
    with pd.ExcelWriter(NOMBRE_ARCHIVO_SALIDA, engine='xlsxwriter') as writer:
        final_df.drop(columns=['Fecha', 'DOY', 'Evaporacion_HS_ET0 [mm]'], inplace=True, errors='ignore')
        final_df.to_excel(writer, sheet_name='Resultados Diarios', index=False)
        df_resumen.to_excel(writer, sheet_name='Resumen General', index=False)
    print(f"\nResultados guardados en '{NOMBRE_ARCHIVO_SALIDA}'.")
except Exception as e:
    print(f"\nERROR: No se pudo guardar el archivo Excel. Detalle del error: {e}")