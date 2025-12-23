import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d, PchipInterpolator
from dataclasses import dataclass

# =============================================================================
# 1. 輔助函數與資料類別定義 (Helper Functions & Data Classes)
# =============================================================================

# --- 全域常數設定 ---
PLOT_WIDTH_SCALE = 0.9 
DEFAULT_FIGSIZE = (10 * PLOT_WIDTH_SCALE, 7 * PLOT_WIDTH_SCALE)

@dataclass
class AnalysisParameters:
    """一個用來儲存所有分析參數的資料類別，方便在函數間傳遞。"""
    s_ds: float
    s_d1: float
    building_type: str
    demand_spectrum_type: str
    initial_damp_ratio: float
    Ca: float
    Cv: float
    T_M: float
    xlim_factor: float
    ylim_factor: float
    damping_factor: float

# <<<< 修正點: 補上遺漏的 log_message 函數定義 >>>>
def log_message(message: str, level: str = 'info'):
    """將訊息同時顯示在當前頁面並記錄到 session_state 中，以便刷新後顯示。"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    # 記錄訊息與其類型 (info, warning, error)
    st.session_state.log_messages.append((message, level))
    
    # 即時顯示 (此訊息會在 rerun 後消失)
    if level == 'info':
        st.info(message)
    elif level == 'warning':
        st.warning(message)
    elif level == 'error':
        st.error(message)

def get_k_factor(building_type: str) -> float:
    """根據建築類型返回對應的 k 係數。"""
    return {"A": 1.0, "B": 0.67, "C": 0.33}.get(building_type, 0.33)

def find_api_and_area_slice(current_dpi, Sa_interp, Sd_interp, min_sd, max_sd):
    """在容量曲線上找到對應api，並返回計算面積所需的數據切片。"""
    if not (min_sd <= current_dpi <= max_sd): return None, [], []
    api = np.interp(current_dpi, Sd_interp, Sa_interp)
    if api < 0: return None, [], []
    
    idx = np.searchsorted(Sd_interp, current_dpi, side='right')
    x_cal = np.concatenate(([0], Sd_interp[:idx], [current_dpi]))
    y_cal = np.concatenate(([0], Sa_interp[:idx], [api]))
    unique_x, indices = np.unique(x_cal, return_index=True)
    return api, unique_x, y_cal[indices]

def generate_ap_curve(capacity_sd, capacity_sa, params: AnalysisParameters, m_stiffness, k_factor):
    """為整條容量曲線，逐點計算對應的Ap曲線，並返回Ap值和偵錯用的DataFrame。"""
    damping_table = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0])
    bs_table = np.array([0.8, 0.8, 1.0, 1.33, 1.6, 1.79, 1.87, 1.93, 1.93])
    b1_table = np.array([0.8, 0.8, 1.0, 1.25, 1.5, 1.63, 1.7, 1.75, 1.75])
    
    ap_curve, debug_data = [], []
    for i in range(len(capacity_sd)):
        sd_i, sa_i = capacity_sd[i], capacity_sa[i]
        if sd_i <= 0 or sa_i <= 0 or m_stiffness <= 0:
            ap_curve.append(sa_i)
            debug_data.append({'Sd (m)': sd_i, 'Sa (g)': sa_i, 'Teq (sec)': 0, 'dmp_eff (%)': 0, 'Bs': 0, 'B1': 0, 'T0 (sec)': 0, 'Ap (g)': sa_i})
            continue
        
        idx = i + 1; x_cal, y_cal = capacity_sd[:idx], capacity_sa[:idx]
        area_0 = np.trapezoid(y_cal, x_cal)
        math_b = sd_i * m_stiffness - sa_i
        dy = (2 * area_0 - sd_i * sa_i) / math_b if abs(math_b) > 1e-9 else 0
        ay = m_stiffness * dy

        dmp_eff = params.initial_damp_ratio
        if dy > 0 and ay > 0:
            dmp_0 = 63.7 * (ay*sd_i - dy*sa_i) / (sd_i*sa_i) if sd_i*sa_i > 1e-9 else 0
            dmp_eff = max(k_factor * dmp_0 + params.initial_damp_ratio, params.initial_damp_ratio)

        damping_ratio_decimal = dmp_eff / 100.0
        Bs = np.interp(damping_ratio_decimal, damping_table, bs_table)
        B1 = np.interp(damping_ratio_decimal, damping_table, b1_table)
        T0 = (params.s_d1 * Bs) / (params.s_ds * B1) if (params.s_ds * B1) > 0 else float('inf')
        Teq = 2 * math.pi * math.sqrt(sd_i / (sa_i * 9.81)) if sa_i > 0 else float('inf')
        
        Ap_i = sa_i
        if Teq <= 0.2 * T0 and Bs > 0 and T0 > 0:
            denominator = (1 + (2.5 / Bs - 1) * Teq / (0.2 * T0))
            if denominator != 0: Ap_i = sa_i / denominator
        elif 0.2 * T0 < Teq <= T0: Ap_i = Bs * sa_i / 2.5
        elif T0 < Teq: Ap_i = (Bs * sa_i / 2.5) * (Teq / T0) if T0 > 0 else sa_i
        
        ap_curve.append(Ap_i)
        debug_data.append({'Sd (m)': sd_i, 'Sa (g)': sa_i, 'Teq (sec)': Teq, 'dmp_eff (%)': dmp_eff, 'Bs': Bs, 'B1': B1, 'T0 (sec)': T0, 'Ap (g)': Ap_i})

    return ap_curve, pd.DataFrame(debug_data)

def get_image_download_link(fig, filename, text):
    """提供Matplotlib圖表的下載連結。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button(label=text, data=buf.getvalue(), file_name=filename, mime="image/jpeg")


# =============================================================================
# 2. Streamlit UI 介面定義 (Interface Definition)
# =============================================================================

def setup_ui():
    """設定整個 Streamlit 頁面的使用者介面，並返回所有輸入資料。"""
    st.set_page_config(page_title="結構分析工具", layout="wide")
    st.title("🏗️ 結構分析工具：容量震譜 與 性能點分析")

    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        st.subheader("請輸入密碼以繼續")
        password_input = st.text_input("密碼", type="password")
        if st.button("登入"):
            if password_input == st.secrets["APP_PASSWORD"]: st.session_state['logged_in'] = True; st.rerun()
            else: st.error("密碼錯誤，請重試。")
        st.stop()
    
    st.sidebar.button("登出", on_click=lambda: st.session_state.update({
        'logged_in': False, 'fig_performance_point': None, 'fig_capacity_curve': None, 
        'fig_ap_curve': None, 'performance_point_results': None, 
        'performance_point_status': None, 'iteration_data': [], 'df_spectrum_for_analysis': None
    }))

    st.header("📄 Step 1：貼上樓層資料")
    st.markdown("請貼上兩欄資料：**Weight** (tf) 與 **Mode Shape (φ)**。")
    floor_data = st.text_area("...", placeholder="100.0\t1\n120.0\t0.89\n115.0\t0.66\n...", height=150, label_visibility="collapsed")
    W, PF1, ALPHA1, df_floor = None, None, None, None
    if floor_data:
        try:
            df_floor = pd.read_csv(io.StringIO(floor_data), sep='\t', header=None, names=['Weight_t', 'Phi'])
            n = len(df_floor)
            floor_names = ['RF'] + [f'{n-i+1}F' if n-i != 1 else '2F' for i in range(1, n)]
            df_floor.insert(0, 'Floor', floor_names)
            col1, col2 = st.columns([2, 1])
            col1.dataframe(df_floor, use_container_width=True)
            W, PF1, ALPHA1 = df_floor['Weight_t'].sum(), 0, 0
            if W > 0 and (df_floor['Weight_t'] * df_floor['Phi']**2).sum() > 0:
                PF1 = (df_floor['Weight_t'] * df_floor['Phi']).sum() / (df_floor['Weight_t'] * df_floor['Phi']**2).sum()
                ALPHA1 = (df_floor['Weight_t'] * df_floor['Phi']).sum() / W * PF1
            col2.metric("🔹 總重量 W (tf)", f"{W:.2f}"); col2.metric("🔹 PF1", f"{PF1:.4f}"); col2.metric("🔹 ALPHA1", f"{ALPHA1:.4f}")
        except Exception as e: st.error(f"❌ 樓層資料讀取失敗: {e}")

    st.header("📄 Step 2：貼上容量曲線資料")
    st.markdown("請貼上兩欄資料：**Displacement** (m) 與 **Base Shear** (tf)。")
    curve_data = st.text_area("...", placeholder="0\t0\n0.1\t130\n0.2\t280\n...", height=150, label_visibility="collapsed")
    df_curve = None
    if curve_data:
        try:
            df_curve = pd.read_csv(io.StringIO(curve_data), sep='\t', header=None, names=['Displacement_m', 'BaseShear_tf'])
            st.dataframe(df_curve, use_container_width=True)
        except Exception as e: st.error(f"❌ 容量曲線資料讀取失敗: {e}")
    
    with st.expander("📌 標示特定性能點 (可選)"):
        special_points = {}
        points_to_define = {"Vy": "初始降伏地震力", "Ve": "設計地震力", "V475": "475年性能點", "V2500": "2500年性能點"}
        for key, name in points_to_define.items():
            st.markdown(f"**{name} ({key})**")
            cols = st.columns(2)
            disp = cols[0].number_input(f"{key} Displacement (m)", key=f"{key}_disp", value=0.0, format="%.4f")
            shear = cols[1].number_input(f"{key} Base Shear (tf)", key=f"{key}_shear", value=0.0, format="%.2f")
            if disp > 0 and shear > 0: special_points[key] = (disp, shear)

    st.header("📄 Step 3：容量震譜")
    st.markdown("請貼上或自動轉換容量震譜資料：**Sd** (m) 與 **Sa** (g)。")
    if 'converted_spectrum_data_str' not in st.session_state: st.session_state.converted_spectrum_data_str = None
    col_generate, _ = st.columns([1, 5])
    with col_generate:
        if st.button("🔁 使用 Step 1+2 自動轉換"):
            if all(v is not None for v in [df_curve, W, PF1, ALPHA1]):
                df_gen = pd.DataFrame()
                if PF1 != 0: df_gen['Sd_m'] = df_curve['Displacement_m'] / PF1
                else: st.error("❌ PF1 為零。"); st.stop()
                if ALPHA1 != 0: df_gen['Sa_g'] = df_curve['BaseShear_tf'] / W / ALPHA1
                else: st.error("❌ ALPHA1 為零。"); st.stop()
                df_gen = df_gen.sort_values(by='Sd_m').drop_duplicates(subset='Sd_m', keep='first').dropna()
                if not df_gen.empty: 
                    st.session_state.converted_spectrum_data_str = df_gen.to_csv(sep='\t', index=False, header=False)
                    st.session_state.df_spectrum_for_analysis = df_gen; st.rerun()
                else: st.warning("⚠️ 轉換結果為空")
            else: st.warning("⚠️ 請先完成 Step 1 和 Step 2")

    initial_text = st.session_state.converted_spectrum_data_str if st.session_state.converted_spectrum_data_str else ""
    spectrum_data = st.text_area("...", value=initial_text, placeholder="0.00\t0.00\n0.05\t0.07\n0.07\t0.13\n...", height=150, label_visibility="collapsed")
    if st.session_state.converted_spectrum_data_str: st.session_state.converted_spectrum_data_str = None
    
    if spectrum_data:
        try:
            df_spectrum_parsed = pd.read_csv(io.StringIO(spectrum_data), sep='\t', header=None, names=['Sd_m', 'Sa_g']).sort_values(by='Sd_m').drop_duplicates(subset='Sd_m', keep='first')
            st.session_state.df_spectrum_for_analysis = df_spectrum_parsed
            if 'df_spectrum_displayed' not in st.session_state or st.session_state.df_spectrum_displayed is None or not st.session_state.df_spectrum_displayed.equals(df_spectrum_parsed):
                st.success("✅ 容量震譜資料已載入"); st.dataframe(df_spectrum_parsed, use_container_width=True)
                st.session_state.df_spectrum_displayed = df_spectrum_parsed
        except Exception as e: st.error(f"❌ 容量震譜讀取失敗: {e}")

    st.header("📄 Step 4：輸入地震需求參數並計算性能點")
    cols = st.columns(5)
    sds = cols[0].number_input("S_DS / S_MS", value=1.0, step=0.05, format="%.3f")
    sd1 = cols[1].number_input("S_D1 / S_M1", value=0.6, step=0.05, format="%.3f")
    b_type = cols[2].selectbox("Building Type", ["A", "B", "C"])
    spec_type = cols[3].selectbox("Spectrum Type", ["Type 1", "Type 2"])
    damp_ratio = cols[4].number_input("固有阻尼比 (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    
    st.markdown("---")
    st.subheader("分析與圖表選項")
    plot_cols = st.columns(3)
    xlim_factor = plot_cols[0].number_input("X軸範圍係數", min_value=1.1, value=3.0, step=0.1)
    ylim_factor = plot_cols[1].number_input("Y軸範圍係數", min_value=1.1, value=3.0, step=0.1)
    damping_factor = plot_cols[2].number_input("阻尼/鬆弛係數", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
    Ca = 0.4 * sds
    params = AnalysisParameters(sds, sd1, b_type, spec_type, damp_ratio, Ca, sd1, sd1/(2.5*Ca) if Ca > 0 else float('inf'), xlim_factor, ylim_factor, damping_factor)

    return W, PF1, ALPHA1, df_curve, params, special_points

# =============================================================================
# 3. 主程式執行邏輯 (Main Execution Logic)
# =============================================================================

def run_analysis(W, PF1, ALPHA1, df_spectrum, params: AnalysisParameters, fig_size: tuple):
    """執行性能點分析的主函數。"""
    log_message(f"依據輸入參數，計算得到 Ca = {params.Ca:.3f}, Cv = {params.Cv:.3f}", level='info')
    if len(df_spectrum) < 2: log_message("❌ 容量震譜資料不足。", level='error'); return
    
    min_sd_raw, max_sd_raw = df_spectrum['Sd_m'].min(), df_spectrum['Sd_m'].max()
    max_sa_raw = df_spectrum['Sa_g'].max()
    
    interp_func = PchipInterpolator(df_spectrum['Sd_m'], df_spectrum['Sa_g'])
    Sd_interp = np.linspace(min_sd_raw, max_sd_raw, 200)
    Sa_interp = interp_func(Sd_interp)
    
    search_range = int(len(Sd_interp) * 0.3)
    if search_range < 2: search_range = min(2, len(Sd_interp))
    secant_stiffnesses = [Sa_interp[i]/Sd_interp[i] for i in range(1, search_range) if Sd_interp[i] > 1e-9]
    if not secant_stiffnesses: m = (Sa_interp[1]-Sa_interp[0])/(Sd_interp[1]-Sa_interp[0]) if len(Sa_interp)>1 and (Sd_interp[1]-Sa_interp[0])>1e-9 else 0
    else: m = np.max(secant_stiffnesses)

    T = np.arange(0.01, 6.5, 0.01)
    
    # ===== 新增：根據初始阻尼比調整需求震譜 =====
    initial_damping_ratio_decimal = params.initial_damp_ratio / 100.0
    damping_table = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0])
    bs_table = np.array([0.8, 0.8, 1.0, 1.33, 1.6, 1.79, 1.87, 1.93, 1.93])
    b1_table = np.array([0.8, 0.8, 1.0, 1.25, 1.5, 1.63, 1.7, 1.75, 1.75])
    
    Bs_initial = np.interp(initial_damping_ratio_decimal, damping_table, bs_table)
    B1_initial = np.interp(initial_damping_ratio_decimal, damping_table, b1_table)
    
    # 計算 T0（短週期與中週期的分界點）
    T0_initial = (params.s_d1 * Bs_initial) / (params.s_ds * B1_initial) if (params.s_ds * B1_initial) > 0 else float('inf')
    
    # 依照週期範圍使用不同的調整係數生成初始需求震譜
    Sa_demand_initial = []
    for t in T:
        if t <= 0.2 * T0_initial:
            # 較短週期：使用 Bs 在分母
            Sa_adjusted = params.s_ds * (0.4 + (1/Bs_initial - 0.4) * t / (0.2 * T0_initial))
        elif 0.2 * T0_initial < t <= T0_initial:
            # 短週期：使用 Bs 在分母
            Sa_adjusted = params.s_ds / Bs_initial
        elif T0_initial < t <= 2.5 * T0_initial:
            # 中週期：使用 B1 在分母
            Sa_adjusted = params.s_d1 / (B1_initial * t)
        else:
            # 長週期：使用 Bs 在分母
            Sa_adjusted = 0.4 * params.s_ds / Bs_initial
        
        Sa_demand_initial.append(Sa_adjusted)
    # ===== 新增結束 =====

    Sd_demand_initial = [(t/(2*math.pi))**2 * sa * 9.81 for t, sa in zip(T, Sa_demand_initial)]
    
    dpi, dpi_found = 0.0, False
    if m > 0 and len(Sd_demand_initial) > 1:
        diff = m * np.array(Sd_demand_initial) - np.array(Sa_demand_initial)
        indices = np.where(np.diff(np.sign(diff)))[0]
        if len(indices) > 0:
            idx = indices[0]; x1, x2 = Sd_demand_initial[idx], Sd_demand_initial[idx+1]; y1, y2 = diff[idx], diff[idx+1]
            if abs(y2 - y1) > 1e-9: dpi = x1 - y1 * (x2 - x1) / (y2 - y1); dpi_found = True

    if dpi_found and dpi > max_sd_raw:
        log_message(f"初始試算點 ({dpi:.3f}m) 超出容量譜範圍，已自動修正為容量譜最大位移點 ({max_sd_raw:.3f}m) 開始迭代。", level='warning')
        dpi = max_sd_raw
    elif not dpi_found:
        log_message("警告：結構剛度與初始需求譜無交點，將從容量譜最大位移點開始迭代。", level='warning')
        dpi = max_sd_raw
        
    log_message(f"Step B: 開始迭代計算...", level='info'); max_iter = 100
    dy_final, ay_final = 0, 0
    all_demand_curves = []
    
    for trial in range(max_iter):
        api, x_slice, y_slice = find_api_and_area_slice(dpi, Sa_interp, Sd_interp, min_sd_raw, max_sd_raw)
        if api is None: log_message(f"❌ 迭代 {trial+1}: 試算點位移超出範圍。", level='error'); st.session_state.performance_point_status="not_found"; break
        area = np.trapezoid(y_slice, x_slice)
        dy = (2*area - dpi*api) / (dpi*m - api) if abs(dpi*m - api) > 1e-9 else 0
        ay = m * dy
        if dy <= 0 or ay <= 0: log_message(f"⚠️ 迭代 {trial+1}: 等效雙線性系統無效。", level='warning'); st.session_state.performance_point_status="not_found"; break
        k = get_k_factor(params.building_type)
        dmp_0 = 63.7 * (ay*dpi - dy*api) / (dpi*api) if dpi*api > 1e-9 else 0
        dmp_eff = max(k * dmp_0 + params.initial_damp_ratio, params.initial_damp_ratio)
        SRa = min((3.21 - 0.68*np.log(dmp_eff))/2.12, 1.0) if dmp_eff > 0 else 1.0
        SRv = min((2.31 - 0.411*np.log(dmp_eff))/1.65, 1.0) if dmp_eff > 0 else 1.0
        T_Mnew = params.T_M * (SRv/SRa) if (params.Ca*SRa) > 0 else float('inf')
        if params.demand_spectrum_type == "Type 1": Sa_new = [2.5*params.Ca*SRa if t<T_Mnew else params.Cv*SRv/t for t in T]
        else: Sa_new = [2.5*params.Ca*SRa if t<T_Mnew else max(params.Ca*SRa, params.Cv*SRv/t) for t in T]
        Sd_new = [(t/(2*math.pi))**2 * sa * 9.81 for t, sa in zip(T, Sa_new)]
        all_demand_curves.append((Sd_new, Sa_new, ["DarkSeaGreen","DimGray","Khaki","LightPink","MediumPurple"][trial % 5]))
        f_cap = interp1d(Sd_interp, Sa_interp, 'linear', fill_value="extrapolate")
        f_dem = interp1d(Sd_new, Sa_new, 'linear', fill_value="extrapolate")
        diffs = f_cap(Sd_interp) - f_dem(Sd_interp)
        indices = np.where(np.diff(np.sign(diffs)))[0]
        x_inter = None
        if len(indices) > 0:
            idx = indices[0]
            if abs(diffs[idx+1] - diffs[idx]) > 1e-9:
                x_inter = Sd_interp[idx] - diffs[idx] * (Sd_interp[idx+1] - Sd_interp[idx]) / (diffs[idx+1] - diffs[idx])
        
        if x_inter is None or not (min_sd_raw <= x_inter <= max_sd_raw): 
            log_message(f"ℹ️ 迭代 {trial+1}: 無法找到有效交點，嘗試增加位移以繼續迭代...", level='info')
            st.session_state.iteration_data.append({'Iteration': trial + 1, 'DPI (m)': f"{dpi:.4f}", 'API (g)': f"{api:.4f}", 'Next DPI (m)': "N/A", 'Error (%)': "No Intersection"})
            dpi = dpi * 1.05
            if dpi >= max_sd_raw:
                log_message(f"⚠️ 嘗試增加位移後已達容量譜終點，迭代中止。", level='warning'); st.session_state.performance_point_status = "not_found"; break
            continue

        y_inter = f_cap(x_inter)
        error = np.sqrt(((x_inter-dpi)/dpi)**2 + ((y_inter-api)/api)**2) * 100 if dpi*api > 1e-6 else float('inf')
        st.session_state.iteration_data.append({'Iteration': trial + 1, 'DPI (m)': f"{dpi:.4f}", 'API (g)': f"{api:.4f}", 'Next DPI (m)': f"{x_inter:.4f}", 'Error (%)': f"{error:.2f}"})
        if error < 5:
            teff = 2*math.pi*math.sqrt(x_inter/(y_inter*9.81)) if y_inter > 0 else 0
            Vb = y_inter*W*ALPHA1 if all(v is not None for v in [W, ALPHA1]) else None
            roof_disp = x_inter * PF1 if PF1 is not None else None
            st.session_state.performance_point_results = {'final_dpi': x_inter, 'final_api': y_inter, 'final_dmp_eff': dmp_eff, 'final_trial': trial + 1, 'teff': teff, 'base_shear': Vb, 'roof_disp': roof_disp}
            st.session_state.performance_point_status = "found"
            log_message(f"✅ 性能點已找到！經過 {trial+1} 次迭代收斂。", level='info')
            dy_final, ay_final = dy, ay
            break
        dpi = dpi * (1 - params.damping_factor) + x_inter * params.damping_factor
        if trial == max_iter - 1: log_message(f"⚠️ 已達最大迭代次數，未收斂。", level='warning'); st.session_state.performance_point_status = "not_found"

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(df_spectrum['Sd_m'], df_spectrum['Sa_g'], 's', alpha=0.5, markersize=5, label='Original Points', color='gray')
    ax.plot(Sd_interp, Sa_interp, label='Capacity Spectrum', color='DodgerBlue', lw=2)
    ax.plot(Sd_demand_initial, Sa_demand_initial, label=f'Initial Demand ({params.demand_spectrum_type})', color='tomato', ls='--')
    for sd, sa, color in all_demand_curves: ax.plot(sd, sa, alpha=0.6, ls='-', color=color)

    if st.session_state.performance_point_status == "found":
        pp = st.session_state.performance_point_results
        ax.plot(pp['final_dpi'], pp['final_api'], 'ro', markersize=10, zorder=10, label=f'Performance Point')
        ax.plot([0, dy_final, pp['final_dpi']], [0, ay_final, pp['final_api']], linestyle=':', color='black', label='Bilinear Capacity Curve')
        ap_y, ap_debug_df = generate_ap_curve(Sd_interp, Sa_interp, params, m, get_k_factor(params.building_type))
        if len(ap_y) == len(Sd_interp): st.session_state.fig_ap_curve_data = (Sd_interp, ap_y)
        st.session_state.ap_debug_dataframe = ap_debug_df
        d_ratio = pp['final_dmp_eff'] / 100.0
        d_table = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0])
        bs_pp = np.interp(d_ratio, d_table, [0.8, 0.8, 1.0, 1.33, 1.6, 1.79, 1.87, 1.93, 1.93])
        b1_pp = np.interp(d_ratio, d_table, [0.8, 0.8, 1.0, 1.25, 1.5, 1.63, 1.7, 1.75, 1.75])
        t0_pp = (params.s_d1 * bs_pp) / (params.s_ds * b1_pp) if (params.s_ds * b1_pp) > 0 else float('inf')
        teq_pp, sa_pp = pp['teff'], pp['final_api']
        ap_pp = sa_pp
        if teq_pp <= 0.2 * t0_pp and bs_pp > 0 and t0_pp > 0:
            denom = (1 + (2.5/bs_pp - 1) * teq_pp/(0.2*t0_pp))
            if denom != 0: ap_pp = sa_pp / denom
        elif 0.2 * t0_pp < teq_pp <= t0_pp: ap_pp = bs_pp * sa_pp / 2.5
        elif t0_pp < teq_pp: ap_pp = (bs_pp * sa_pp / 2.5) * (teq_pp / t0_pp) if t0_pp > 0 else sa_pp
        pp.update({'ap': ap_pp, 'bs': bs_pp, 'b1': b1_pp, 't0': t0_pp})
        max_y_from_initial_demand = np.max(Sa_demand_initial) if len(Sa_demand_initial) > 0 else 0
        max_y_data = max(max_sa_raw, max_y_from_initial_demand)
        max_x_user, max_y_user = pp['final_dpi'] * params.xlim_factor, pp['final_api'] * params.ylim_factor
        final_max_x, final_max_y = max(max_x_user, max_sd_raw) * 1.05, max(max_y_user, max_y_data) * 1.05
        ax.set_xlim(left=0, right=final_max_x); ax.set_ylim(bottom=0, top=final_max_y)
    else:
        max_y_from_initial_demand = np.max(Sa_demand_initial) if len(Sa_demand_initial) > 0 else 0
        final_max_x, final_max_y = max_sd_raw * 1.2, max(max_sa_raw, max_y_from_initial_demand) * 1.2
        ax.set_xlim(left=0, right=final_max_x); ax.set_ylim(bottom=0, top=final_max_y)

    ax.set_xlabel('Spectral Displacement Sd (m)'); ax.set_ylabel('Spectral Acceleration Sa (g)')
    ax.set_title(f"Capacity & Demand Spectrum ({params.demand_spectrum_type})"); ax.grid(True); ax.legend(loc='best')
    st.session_state.fig_performance_point = fig

# =============================================================================
# 4. 顯示結果 (Display Results)
# =============================================================================

def display_results(df_curve, special_points, fig_size: tuple):
    """根據 session_state 中的分析結果，顯示所有圖表和數據。"""
    st.markdown("---") 
    st.header("📊 分析結果")
    
    if st.session_state.get('log_messages'):
        st.subheader("📝 分析過程日誌")
        for msg, level in st.session_state.get('log_messages', []):
            if level == 'info': st.info(msg)
            elif level == 'warning': st.warning(msg)
            elif level == 'error': st.error(msg)
    
    results = st.session_state.get('performance_point_results', {})
    if st.session_state.get('performance_point_status') == "found" and results:
        st.subheader("💡 性能點分析結果")
        col1, col2 = st.columns(2)
        col1.markdown(f"**性能點位移 (Sd):** `{results.get('final_dpi', 0):.3f} m`")
        col1.markdown(f"**性能點加速度 (Sa):** `{results.get('final_api', 0):.3f} g`")
        roof_disp_val = results.get('roof_disp')
        if roof_disp_val is not None: col1.markdown(f"**對應屋頂位移 (Dr):** `{roof_disp_val:.3f} m`")
        else: col1.markdown(f"**對應屋頂位移 (Dr):** `無法計算`")
        base_shear_val = results.get('base_shear')
        if base_shear_val is not None: col1.markdown(f"**對應基底剪力 (Vb):** `{base_shear_val:.2f} tf`")
        else: col1.markdown(f"**對應基底剪力 (Vb):** `無法計算`")
        col2.markdown(f"**等效週期 (Teff):** `{results.get('teff', 0):.2f} sec`")
        col2.markdown(f"**有效阻尼比 (dmp_eff):** `{results.get('final_dmp_eff', 0):.2f}%`")
        col2.markdown(f"**性能點加速度 (Ap):** `{results.get('ap', 0):.3f} g`")

    if st.session_state.get('iteration_data'):
        with st.expander("🔄 點擊查看迭代過程數據"):
            if isinstance(st.session_state.get('iteration_data'), list) and st.session_state.get('iteration_data'):
                st.dataframe(pd.DataFrame(st.session_state['iteration_data']), use_container_width=True)
            else: st.write("沒有可顯示的迭代數據。")

    st.markdown("<br>", unsafe_allow_html=True) 
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("📉 容量曲線")
        if df_curve is not None and not df_curve.empty:
            fig1, ax1 = plt.subplots(figsize=fig_size)
            ax1.plot(df_curve['Displacement_m'], df_curve['BaseShear_tf'], marker='o', color='green', lw=2, label='Capacity Curve')
            if special_points:
                for name, (disp, shear) in special_points.items():
                    ax1.plot(disp, shear, 'x', color='red', markersize=10, mew=2, label=f'{name} Point')
                    ax1.text(disp, shear, f"  {name}\n  ({disp:.3f}, {shear:.0f})", verticalalignment='top', fontsize=9)
            ax1.set_xlabel("Displacement (m)"); ax1.set_ylabel("Base Shear (tf)")
            ax1.set_title("Capacity Curve"); ax1.grid(True, ls='--', alpha=0.6); ax1.legend()
            st.pyplot(fig1, use_container_width=True)
            get_image_download_link(fig1, "capacity_curve.jpg", "⬇️ 下載容量曲線 (JPG)")
        else: st.info("⚠️ 請在 Step 2 貼上資料以繪圖。")

    with row1_col2:
        st.subheader("📈 容量震譜與性能點")
        if st.session_state.get('fig_performance_point'): 
            st.pyplot(st.session_state['fig_performance_point'], use_container_width=True)
            get_image_download_link(st.session_state['fig_performance_point'], "performance_point.jpg", "⬇️ 下載性能點圖 (JPG)")
        else: st.info("點擊上方按鈕來生成此圖。")

    row2_col1, _ = st.columns(2)
    with row2_col1:
        st.subheader("✨ 性能曲線 (Ap-Sd)")
        if st.session_state.get('fig_ap_curve_data'):
            sd_vals, ap_vals = st.session_state['fig_ap_curve_data']
            results = st.session_state.get('performance_point_results', {})
            fig_ap, ax_ap = plt.subplots(figsize=fig_size)
            ax_ap.plot(sd_vals, ap_vals, color='purple', lw=2, label='Ap Curve')
            if results:
                ax_ap.plot(results.get('final_dpi',0), results.get('ap',0), 'ro', markersize=10, label=f"Performance Point Ap = {results.get('ap',0):.3f}g")
            ax_ap.set_xlabel("Spectral Displacement Sd (m)"); ax_ap.set_ylabel("Performance Acceleration Ap (g)")
            ax_ap.set_title("Performance Curve (Ap-Sd)"); ax_ap.grid(True, ls='--', alpha=0.6); ax_ap.legend()
            st.pyplot(fig_ap, use_container_width=True)
            get_image_download_link(fig_ap, "ap_curve.jpg", "⬇️ 下載性能曲線 (JPG)")
            
            if st.session_state.get('ap_debug_dataframe') is not None:
                with st.expander("🔬 點擊查看性能曲線計算過程"):
                    st.dataframe(st.session_state.get('ap_debug_dataframe').style.format("{:.4f}"))
        else: st.info("找到性能點後將在此處繪製性能曲線。")
    
# =============================================================================
# 5. 主函數 (Main Function)
# =============================================================================

def main():
    """主函數，組織UI和分析流程。"""
    for key in ['logged_in', 'converted_spectrum_data_str', 'df_spectrum_for_analysis', 'df_spectrum_displayed', 'log_messages']:
        if key not in st.session_state: st.session_state[key] = None
    
    W, PF1, ALPHA1, df_curve, params, special_points = setup_ui()

    if st.button("📈 Plot Capacity Spectrum & Performance Point"):
        st.session_state.update({'performance_point_status': None, 'performance_point_results': None, 
                                 'fig_performance_point': None, 'fig_ap_curve_data': None, 
                                 'fig_ap_curve': None, 'iteration_data': [], 'ap_debug_dataframe': None,
                                 'log_messages': []})
        
        df_spectrum_to_analyze = st.session_state.get('df_spectrum_for_analysis')
        if df_spectrum_to_analyze is not None and not df_spectrum_to_analyze.empty:
            run_analysis(W, PF1, ALPHA1, df_spectrum_to_analyze, params, DEFAULT_FIGSIZE)
        else:
            log_message("⚠️ 請先在 Step 3 輸入或轉換資料。", level='warning')
        st.rerun()

    display_results(df_curve, special_points, DEFAULT_FIGSIZE)

if __name__ == "__main__":
    main()
