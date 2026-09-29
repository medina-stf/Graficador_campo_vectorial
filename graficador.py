import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

st.set_page_config(page_title="Análisis de Gradientes y Campos Vectoriales", layout="wide")

# Manejo de memoria
if 'ecuacion' not in st.session_state:
    st.session_state.ecuacion = "sin(x) * cos(y)"
if 'ecuacion_graficada' not in st.session_state:
    st.session_state.ecuacion_graficada = "sin(x) * cos(y)"

def agregar_texto(simbolo):
    st.session_state.ecuacion += simbolo

def borrar_ultimo():
    if len(st.session_state.ecuacion) > 0:
        st.session_state.ecuacion = st.session_state.ecuacion[:-1]

def limpiar_todo():
    st.session_state.ecuacion = ""

def ejecutar_grafica():
    st.session_state.ecuacion_graficada = st.session_state.ecuacion

st.title("📈 Graficadora de Campos Vectoriales (2D y 3D)")
st.markdown("Herramienta interactiva para el análisis de funciones multivariables, curvas de nivel y gradientes.")

# Barra lateral
with st.sidebar:
    st.header("⚙️ Controles de Visualización")
    
    modo_vista = st.radio("Selecciona la dimensión:", ["2D (Vista Superior)", "3D (Cuerpo y Relieve)"], index=1)
    
    st.write("¿Qué elementos deseas mostrar?")
    mostrar_superficie = st.checkbox("Superficie / Curvas de Nivel", value=True)
    mostrar_vectores = st.checkbox("Campo Vectorial (Flechas)", value=True)
    
    st.divider()
    
    # Controles de color
    st.write("Estilo de Vectores")
    estilo_color = st.radio("Modo de color:", ["Sólido (Cumbres y Valles)", "Degradado (Magnitud del Gradiente)"])
    
    color_valles = "#0000FF" # Azul por defecto
    color_cumbres = "#FF0000" # Rojo por defecto
    
    if estilo_color == "Sólido (Cumbres y Valles)":
        col1, col2 = st.columns(2)
        color_valles = col1.color_picker("Color Valles", "#0000FF")
        color_cumbres = col2.color_picker("Color Cumbres", "#FF0000")
        
    st.divider()
    
    ecuacion_str = st.text_input("Función f(x,y):", key="ecuacion")
    st.button("Graficar Función", on_click=ejecutar_grafica, type="primary", use_container_width=True)
    
    with st.expander("Teclado Matemático"):
        # Teclado ampliado con SymPy
        teclas = [
            [('x', 'x'), ('y', 'y'), ('e', 'E'), ('π', 'pi')],
            [('(', '('), (')', ')'), ('|x|', 'Abs('), ('√', 'sqrt(')],
            [('sin', 'sin('), ('cos', 'cos('), ('tan', 'tan('), ('exp', 'exp(')],
            [('asin', 'asin('), ('acos', 'acos('), ('atan', 'atan('), ('ln', 'ln(')],
            [('sinh', 'sinh('), ('cosh', 'cosh('), ('tanh', 'tanh('), ('log', 'log(')],
            [('7', '7'), ('8', '8'), ('9', '9'), ('÷', '/')],
            [('4', '4'), ('5', '5'), ('6', '6'), ('×', '*')],
            [('1', '1'), ('2', '2'), ('3', '3'), ('-', '-')],
            [('0', '0'), ('.', '.'), ('xⁿ', '**('), ('+', '+')] 
        ]
        for fila_idx, fila in enumerate(teclas):
            cols = st.columns(4)
            for i, (etiqueta, valor) in enumerate(fila):
                cols[i].button(etiqueta, on_click=agregar_texto, args=(valor,), use_container_width=True, key=f"btn_{fila_idx}_{i}")
        
        cols_control = st.columns(2)
        cols_control[0].button("⌫ Borrar", on_click=borrar_ultimo, use_container_width=True)
        cols_control[1].button("C Limpiar", on_click=limpiar_todo, use_container_width=True)

    st.divider()
    st.write("Ajustes de la Gráfica")
    rango_ejes = st.slider("Rango de los ejes (Zoom XY):", 1.0, 100.0, 5.0)
    densidad = st.slider("Densidad de vectores/flechas:", 10, 60, 20)
    
    limite_z = 5.0
    if "3D" in modo_vista:
        limite_z = st.slider("Límite de Altura (Recorte Eje Z):", 1.0, 100.0, 5.0)

# Lógica matemática
if st.session_state.ecuacion_graficada:
    if not mostrar_superficie and not mostrar_vectores:
        st.warning("⚠️ Selecciona al menos una opción (Superficie o Campo Vectorial) en la barra lateral para visualizar la gráfica.")
        st.stop()
        
    try:
        x, y = sp.symbols('x y')
        transformaciones = standard_transformations + (implicit_multiplication_application,)
        f_expr = parse_expr(st.session_state.ecuacion_graficada, transformations=transformaciones)
        
        df_dx_expr = sp.diff(f_expr, x)
        df_dy_expr = sp.diff(f_expr, y)
        
        st.subheader("Cálculos analíticos")
        st.latex(rf"f(x,y) = {sp.latex(f_expr)}")
        st.latex(rf"\nabla f = \left\langle {sp.latex(df_dx_expr)}, \quad {sp.latex(df_dy_expr)} \right\rangle")

        f_num = sp.lambdify((x, y), f_expr, 'numpy')
        df_dx_num = sp.lambdify((x, y), df_dx_expr, 'numpy')
        df_dy_num = sp.lambdify((x, y), df_dy_expr, 'numpy')

        limite = float(rango_ejes)
        x_vals = np.linspace(-limite, limite, densidad)
        y_vals = np.linspace(-limite, limite, densidad)
        X, Y = np.meshgrid(x_vals, y_vals)

        try:
            Z = f_num(X, Y)
            U = df_dx_num(X, Y)
            V = df_dy_num(X, Y)
            
            if np.isscalar(Z): Z = np.full_like(X, Z)
            if np.isscalar(U): U = np.full_like(X, U)
            if np.isscalar(V): V = np.full_like(Y, V)

            # Saneamiento de datos
            Z = np.asarray(Z, dtype=float)
            U = np.asarray(U, dtype=float)
            V = np.asarray(V, dtype=float)
            
            mascara_valida = np.isfinite(Z) & np.isfinite(U) & np.isfinite(V)
            
            Z = np.where(mascara_valida, Z, np.nan)
            U = np.where(mascara_valida, U, np.nan)
            V = np.where(mascara_valida, V, np.nan)
            
            # Acote de crecimiento exponencial
            U = np.clip(U, -50, 50)
            V = np.clip(V, -50, 50)
            
            # Acotar Z para asíntotas
            Z = np.clip(Z, -limite_z * 2.0, limite_z * 2.0)
            
            # Calcular la magnitud del gradiente
            magnitud = np.sqrt(U**2 + V**2)

        except Exception:
            st.error("Hubo un problema evaluando la matriz numérica. Revisa si el rango genera divisiones por cero.")
            st.stop()

        # Renderizado según el modo (2D o 3D)
        if "2D" in modo_vista:
            fig = plt.figure(figsize=(10, 8))
            fig.subplots_adjust(bottom=0.15)
            
            fig.text(0.95, 0.02, 'ISC 3A', fontsize=12, color='gray', 
                     ha='right', va='bottom', style='italic', weight='bold')

            if mostrar_superficie:
                plt.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.4)
            
            if mostrar_vectores:
                if estilo_color == "Degradado (Magnitud del Gradiente)":
                    flechas = plt.quiver(X, Y, U, V, magnitud, cmap='jet', scale=None, headwidth=4)
                    if np.nanmin(magnitud) != np.nanmax(magnitud):
                        cbar = plt.colorbar(flechas)
                        cbar.set_label('Magnitud del Gradiente ||∇f||')
                else:
                    # Modo sólido 2D usando los colores elegidos
                    colores_solidos = ListedColormap([color_valles, color_cumbres])
                    # Se crea un arreglo binario para asegurar un corte perfecto a la mitad
                    umbral = (np.nanmax(Z) + np.nanmin(Z)) / 2.0
                    Z_binario = np.where(Z >= umbral, 1, 0)
                    flechas = plt.quiver(X, Y, U, V, Z_binario, cmap=colores_solidos, scale=None, headwidth=4)
                    
                    # Leyenda personalizada para el modo sólido
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=color_cumbres, label='Cumbres'),
                                       Patch(facecolor=color_valles, label='Valles')]
                    plt.legend(handles=legend_elements, loc='upper right')

            plt.title('Mapa del Gradiente y Curvas de Nivel')
            plt.xlabel('Eje X')
            plt.ylabel('Eje Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.axhline(0, color='black', linewidth=1)
            plt.axvline(0, color='black', linewidth=1)

            st.pyplot(fig)

        elif "3D" in modo_vista:
            fig = go.Figure()
            
            piso = max(np.nanmin(Z) - (limite_z * 0.05), -limite_z)
            Z_piso = np.full_like(Z.flatten(), piso)
            Z_flat = Z.flatten()
            umbral = (np.nanmax(Z) + np.nanmin(Z)) / 2.0
            
            mask_valles = Z_flat < umbral
            mask_cumbres = Z_flat >= umbral

            if mostrar_superficie:
                fig.add_trace(go.Surface(
                    x=x_vals, y=y_vals, z=Z,
                    colorscale='Greys', opacity=0.6,
                    showscale=False, name='Topografía', showlegend=True
                ))

            if mostrar_vectores:
                if estilo_color == "Degradado (Magnitud del Gradiente)":
                    fig.add_trace(go.Cone(
                        x=X.flatten(), y=Y.flatten(), z=Z_piso.flatten(),
                        u=U.flatten(), v=V.flatten(), w=np.zeros_like(Z_flat),
                        colorscale='Jet', sizemode="scaled", sizeref=0.5,
                        showscale=True, name='Magnitud Vectorial', showlegend=False,
                        colorbar=dict(title="||∇f||", len=0.5, x=0.9)
                    ))
                else:
                    if np.any(mask_valles):
                        fig.add_trace(go.Cone(
                            x=X.flatten()[mask_valles], y=Y.flatten()[mask_valles], z=Z_piso[mask_valles],
                            u=U.flatten()[mask_valles], v=V.flatten()[mask_valles], w=np.zeros_like(Z_flat)[mask_valles],
                            colorscale=[[0, color_valles], [1, color_valles]], sizemode="scaled", sizeref=0.5,
                            showscale=False, name='Valles (Dirección del gradiente)', showlegend=True
                        ))
                    if np.any(mask_cumbres):
                        fig.add_trace(go.Cone(
                            x=X.flatten()[mask_cumbres], y=Y.flatten()[mask_cumbres], z=Z_piso[mask_cumbres],
                            u=U.flatten()[mask_cumbres], v=V.flatten()[mask_cumbres], w=np.zeros_like(Z_flat)[mask_cumbres],
                            colorscale=[[0, color_cumbres], [1, color_cumbres]], sizemode="scaled", sizeref=0.5,
                            showscale=False, name='Cumbres (Dirección del gradiente)', showlegend=True
                        ))

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(
                    title="<b>Simbología:</b>",
                    yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1
                ),
                scene=dict(
                    xaxis_title='Eje X', yaxis_title='Eje Y', zaxis_title='Altura Z',
                    zaxis=dict(range=[-limite_z, limite_z]), 
                    aspectratio=dict(x=1, y=1, z=0.6)
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<p style='text-align: right; color: gray; font-style: italic; font-weight: bold;'>ISC 3A</p>", unsafe_allow_html=True)

    except sp.SympifyError:
        st.error("Error de sintaxis matemática. Revisa que los paréntesis estén cerrados.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la gráfica: {type(e).__name__}: {e}")