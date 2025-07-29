import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d

# --- Streamlit 頁面設定 ---
st.set_page_config(page_title="結構分析工具", layout="wide")
st.title("🏗️ 結構分析工具：容量震譜 與 性能點分析")

# --- 簡單密碼檢查功能 ---
# 設定你的密碼
CORRECT_PASSWORD = "chihwei" # <<<<<<< 請在這裡更改你的密碼！

# 從 Streamlit 的 session_state 中獲取登入狀態，如果沒有則預設為 False
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 如果尚未登入，則顯示密碼輸入框
if not st.session_state['logged_in']:
    st.subheader("請輸入密碼以繼續")
    password_input = st.text_input("密碼", type="password")
    
    if st.button("登入"):
        if password_input == CORRECT_PASSWORD:
            st.session_state['logged_in'] = True
            st.success("登入成功！")
            st.rerun() # 登入成功後重新運行，顯示應用程式內容
        else:
            st.error("密碼錯誤，請重試。")
    st.stop() # 如果未登入或密碼錯誤，停止執行應用程式的其餘部分

# 如果已經登入，則顯示登出按鈕 (可選)
st.sidebar.button("登出", on_click=lambda: st.session_state.update({'logged_in': False, 'fig_performance_point': None, 'fig_capacity_curve': None, 'performance_point_results': None}))
# 登出時清空 session_state 中的圖表和結果，確保下次登入時是乾淨的狀態

# --- 通用設定：調整圖表尺寸 ---
PLOT_WIDTH_SCALE = 0.9 # 調整為 0.9 或 1.0 都可以試試看
DEFAULT_FIGSIZE = (10 * PLOT_WIDTH_SCALE, 7 * PLOT_WIDTH_SCALE)

# --- 輔助函數：將圖表保存為 JPG 並提供下載 ---
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label=text,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/jpeg"
    )

# -------------------------
# Step 1：樓層資料
# -------------------------
st.header("📄 Step 1：貼上樓層資料")
st.markdown("請貼上 **樓層重量與主控模態 φ**，用 Excel 貼上或手動輸入。")

floor_data = st.text_area(
    "樓層資料貼上區",
    placeholder="100\t1\n150\t0.8\n150\t0.6\n180\t0.3",
    height=200
)

W = PF1 = ALPHA1 = None
df_floor = None
if floor_data:
    try:
        df_floor = pd.read_csv(io.StringIO(floor_data), sep='\t', header=None, names=['Weight_t', 'Phi'])
        n = len(df_floor)
        floor_names = []
        if n > 0:
            floor_names.append('RF')
            if n > 1:
                for i in range(1, n):
                    if n - i == 1:
                        floor_names.append('2F')
                    else:
                        floor_names.append(f'{n-i+1}F')
        
        df_floor.insert(0, 'Floor', floor_names)
        st.success("✅ Floor data loaded successfully")
        col1, col2 = st.columns([2, 1])
        col1.dataframe(df_floor, use_container_width=True)

        W = df_floor['Weight_t'].sum()
        PF1 = (df_floor['Weight_t'] * df_floor['Phi']).sum() / (df_floor['Weight_t'] * df_floor['Phi']**2).sum()
        ALPHA1 = (df_floor['Weight_t'] * df_floor['Phi']).sum() / W *PF1

        col2.metric("🔹 Total Structure Weight W (t)", f"{W:.2f}")
        col2.metric("🔹 PF1", f"{PF1:.4f}")
        col2.metric("🔹 ALPHA1", f"{ALPHA1:.4f}")
    except Exception as e:
        st.error(f"❌ Failed to load floor data: {e}")


# -------------------------
# Step 2：容量曲線資料
# -------------------------
st.header("📄 Step 2：貼上容量曲線資料")
st.markdown("請貼上 **位移（m）與 Base Shear（tf）**")

curve_data = st.text_area(
    "容量曲線資料貼上區",
    placeholder="0\t0\n0.1\t1000\n0.2\t1500\n0.3\t2000",
    height=200
)

df_curve = None
if curve_data:
    try:
        df_curve = pd.read_csv(io.StringIO(curve_data), sep='\t', header=None, names=['Displacement_m', 'BaseShear_tf'])
        st.success("✅ Capacity curve data loaded successfully")
        st.dataframe(df_curve, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to load curve data: {e}")


# -------------------------
# Step 3：容量震譜資料（可手動或自動產生）
# -------------------------
st.header("📄 Step 3：容量震譜 Sd-Sa")
st.markdown("您可以選擇貼上容量震譜資料，或使用 Step 1 + Step 2 自動轉換。")

spectrum_data = st.text_area(
    "容量震譜資料貼上區（可跳過）",
    placeholder="0.0\t0.0\n0.1\t0.2\n0.2\t0.35\n0.3\t0.4",
    height=200
)

col_generate, _ = st.columns([1, 5])
df_spectrum = None
if col_generate.button("🔁 Auto-convert using Step 1 + Step 2"):
    if df_curve is not None and W is not None and PF1 is not None and ALPHA1 is not None:
        df_generated_spectrum = pd.DataFrame()
        df_generated_spectrum['Sd_m'] = df_curve['Displacement_m'] / PF1
        if ALPHA1 != 0:
            df_generated_spectrum['Sa_g'] = df_curve['BaseShear_tf'] / W / ALPHA1
        else:
            st.error("❌ ALPHA1 is zero, cannot calculate Capacity Spectrum Sa. Please check floor data.")
            df_generated_spectrum['Sa_g'] = np.nan 
        
        df_generated_spectrum.dropna(inplace=True)

        if not df_generated_spectrum.empty:
            spectrum_data = df_generated_spectrum.to_csv(sep='\t', index=False, header=False)
            df_spectrum = df_generated_spectrum
            st.success("✅ Capacity Spectrum successfully converted.")
            st.dataframe(df_generated_spectrum, use_container_width=True)
        else:
            st.warning("⚠️ Converted capacity spectrum data is empty. Please check input data.")
    else:
        st.warning("⚠️ Please complete Step 1 and Step 2 first.")
elif spectrum_data:
    try:
        df_spectrum = pd.read_csv(io.StringIO(spectrum_data), sep='\t', header=None, names=['Sd_m', 'Sa_g'])
        st.success("✅ Capacity Spectrum data successfully loaded.")
        st.dataframe(df_spectrum, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to load Capacity Spectrum data: {e}")


# -------------------------
# Step 4：輸入需求譜參數並計算性能點
# -------------------------
st.header("📄 Step 4：輸入地震需求參數並計算性能點")
col_ca, col_cv, col_type, col_demand_type = st.columns(4)
Ca = col_ca.number_input("Ca", value=0.4, step=0.05)
Cv = col_cv.number_input("Cv", value=0.6, step=0.05)
building_type = col_type.selectbox("Building Type (A/B/C)", ["A", "B", "C"])
demand_spectrum_type = col_demand_type.selectbox("Demand Spectrum Type", ["Type 1", "Type 2"]) 

# Initialize session state for figures and performance point results if not already present
if 'fig_performance_point' not in st.session_state:
    st.session_state['fig_performance_point'] = None
if 'fig_capacity_curve' not in st.session_state:
    st.session_state['fig_capacity_curve'] = None
if 'performance_point_results' not in st.session_state:
    st.session_state['performance_point_results'] = None

if st.button("📌 Plot Capacity Spectrum & Performance Point"):
    if df_spectrum is not None and not df_spectrum.empty:
        # --- 輔助函數定義 ---
        def Interpolate_data(x, y, n_points):
            unique_x, unique_indices = np.unique(x, return_index=True)
            unique_y = np.array(y)[unique_indices]

            if len(unique_x) < 2:
                st.error("Insufficient data points for Capacity Spectrum interpolation. At least two points are required.")
                return [], []

            fx = interp1d(unique_x, unique_y, kind='linear', fill_value="extrapolate")
            x_interp = np.linspace(min(unique_x), max(unique_x), n_points)
            y_interp = fx(x_interp)
            return list(x_interp), list(y_interp)

        def find_api(current_dpi, Sa_capacity_interp, Sd_capacity_interp):
            f_capacity = interp1d(Sd_capacity_interp, Sa_capacity_interp, kind='linear', fill_value="extrapolate")
            api = f_capacity(current_dpi)

            idx = np.searchsorted(Sd_capacity_interp, current_dpi)
            
            x_cal = Sd_capacity_interp[:idx]
            y_cal = Sa_capacity_interp[:idx]

            if current_dpi not in x_cal and current_dpi >= Sd_capacity_interp[0]:
                x_cal.append(current_dpi)
                y_cal.append(api)
            
            sorted_points = sorted(zip(x_cal, y_cal))
            x_cal_sorted = [p[0] for p in sorted_points]
            y_cal_sorted = [p[1] for p in sorted_points]

            return api, x_cal_sorted, y_cal_sorted

        def get_k(building_type, dmp_0, dy, ay, dpi, api):
            # 確保分母不為零，且只在需要時進行複雜計算
            if dpi * api == 0:
                return 0.33 
            
            # 修正 mu 計算，確保 dy > 0
            if dy <= 0:
                mu = 1.0 # 避免除以零或負數
            else:
                mu = dpi / dy
            
            # 此處的 k 值判斷邏輯是簡化，若有更精確的公式，可在此調整
            if building_type == "A":
                return 1.0 
            elif building_type == "B":
                return 0.67
            elif building_type == "C":
                return 0.33 
            return 0.33 # Default fallback

        # --- 開始計算 ---
        Sd_capacity_interp, Sa_capacity_interp = Interpolate_data(df_spectrum['Sd_m'], df_spectrum['Sa_g'], 500)

        if not Sd_capacity_interp or not Sa_capacity_interp:
            st.error("❌ Capacity Spectrum interpolation failed. Please check original data.")
            st.stop()

        T_M = Cv / (2.5 * Ca)
        T = np.arange(0.01, 6.5, 0.01) 

        # 根據需求震譜類型計算 Sa_demand_initial
        if demand_spectrum_type == "Type 1":
            Sa_demand_initial = [2.5 * Ca if t < T_M else Cv / t for t in T]
        elif demand_spectrum_type == "Type 2":
            Sa_demand_initial = []
            for t in T:
                if t < T_M:
                    Sa_demand_initial.append(2.5 * Ca)
                else:
                    Sa_demand_initial.append(max(Ca, Cv / t)) 

        Sd_demand_initial = [(t / (2 * math.pi))**2 * sa * 9.81 for t, sa in zip(T, Sa_demand_initial)]

        # 調整圖表大小
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.plot(Sd_capacity_interp, Sa_capacity_interp, label='Capacity Spectrum', color='DodgerBlue')
        ax.plot(Sd_demand_initial, Sa_demand_initial, label='Initial Demand Spectrum', color='tomato')

        if len(Sd_capacity_interp) < 2:
            st.error("❌ Insufficient Capacity Spectrum data points to calculate initial stiffness.")
            st.stop()

        m = (Sa_capacity_interp[1] - Sa_capacity_interp[0]) / (Sd_capacity_interp[1] - Sd_capacity_interp[0])
        initial_dpi_found = False
        dpi = 0.0 

        if Sd_demand_initial:
            for i in range(len(Sd_demand_initial)):
                if Sd_demand_initial[i] > 0 and m * Sd_demand_initial[i] > Sa_demand_initial[i]:
                    if i > 0:
                        sd1, sa_dem1 = Sd_demand_initial[i-1], Sa_demand_initial[i-1]
                        sd2, sa_dem2 = Sd_demand_initial[i], Sa_demand_initial[i]

                        diff1 = m * sd1 - sa_dem1
                        diff2 = m * sd2 - sa_dem2

                        if diff1 * diff2 < 0 and (diff2 - diff1) != 0: 
                            dpi = sd1 - diff1 * (sd2 - sd1) / (diff2 - diff1)
                            initial_dpi_found = True
                            break
                    if not initial_dpi_found: 
                        dpi = Sd_demand_initial[i]
                        initial_dpi_found = True
                        break

        if not initial_dpi_found or dpi <= 0:
            st.error("❌ Failed to find initial trial point. Please check capacity and demand spectrum data.")
            st.stop()

        final_performance_point = None
        line_colors = ["DarkSeaGreen","DimGray","Khaki","LightPink","MediumPurple"]
        color_idx = 0

        max_iterations = 100 
        st.info(f"Starting iteration to find performance point (max {max_iterations} iterations).")

        for trial in range(max_iterations):
            current_color = line_colors[color_idx % len(line_colors)]
            color_idx += 1

            api, x_cal_for_area, y_cal_for_area = find_api(dpi, Sa_capacity_interp, Sd_capacity_interp)

            if len(x_cal_for_area) < 2 or len(y_cal_for_area) < 2:
                st.warning(f"Iteration {trial+1}: Insufficient points for area calculation. Check capacity spectrum data.")
                break 

            area_0 = np.trapz(y_cal_for_area, x_cal_for_area)

            if m == 0:
                st.error(f"Iteration {trial+1}: Initial slope m is zero, cannot calculate dy and ay.")
                break

            math_b = dpi * m - api
            if abs(math_b) < 1e-9: 
                st.warning(f"Iteration {trial+1}: math_b is close to zero, cannot calculate dy.")
                break
            math_c = 2 * area_0 - dpi * api
            dy = math_c / math_b
            ay = m * dy

            if dy <= 0:
                st.warning(f"Iteration {trial+1}: Calculated dy is non-positive ({dy:.4f}m). This might indicate an inappropriate model or abnormal data.")
                break 

            if dpi * api == 0:
                 dmp_0 = 0 
            else:
                 dmp_0 = 63.7 * (ay*dpi - dy*api) / (dpi*api)
            
            k = get_k(building_type, dmp_0, dy, ay, dpi, api)
            dmp_eff = k * dmp_0 + 5

            if dmp_eff <= 0:
                st.warning(f"Iteration {trial+1}: Effective damping ratio dmp_eff is non-positive ({dmp_eff:.2f}%), cannot calculate reduction factors.")
                break
            
            SRa = (3.21 - 0.68 * np.log(dmp_eff)) / 2.12
            SRv = (2.31 - 0.411 * np.log(dmp_eff)) / 1.65

            if SRa <= 0 or SRv <= 0:
                st.warning(f"Iteration {trial+1}: Reduction factors SRa or SRv are non-positive, might result in an invalid demand spectrum.")
                break
            
            T_Mnew = SRv / SRa * T_M

            if demand_spectrum_type == "Type 1":
                Sa_new = [2.5 * Ca * SRa if t < T_Mnew else SRv * Cv / t for t in T]
            elif demand_spectrum_type == "Type 2":
                Sa_new = []
                for t in T:
                    if t < T_Mnew:
                        Sa_new.append(2.5 * Ca * SRa)
                    else:
                        Sa_new.append(max(Ca * SRa, SRv * Cv / t)) 

            Sd_new = [(t / (2 * math.pi))**2 * sa * 9.81 for t, sa in zip(T, Sa_new)]

            ax.plot(Sd_new, Sa_new, alpha=0.7, linestyle='-')

            x_inter = None
            y_inter = None
            
            min_sd_search = max(Sd_capacity_interp[0], Sd_new[0])
            max_sd_search = min(Sd_capacity_interp[-1], Sd_new[-1])
            
            if min_sd_search >= max_sd_search: 
                st.warning(f"Iteration {trial+1}: Capacity and demand spectrum displacement ranges do not overlap, cannot find intersection.")
                break

            f_capacity_interp_func = interp1d(Sd_capacity_interp, Sa_capacity_interp, kind='linear', fill_value="extrapolate")
            f_demand_interp_func = interp1d(Sd_new, Sa_new, kind='linear', fill_value="extrapolate")

            test_sds = np.linspace(min_sd_search, max_sd_search, 1000)
            diffs = f_capacity_interp_func(test_sds) - f_demand_interp_func(test_sds)

            sign_changes = np.where(np.diff(np.sign(diffs)))[0]

            if len(sign_changes) > 0:
                idx = sign_changes[0] 
                sd1, sd2 = test_sds[idx], test_sds[idx+1]
                diff1, diff2 = diffs[idx], diffs[idx+1]

                if (diff2 - diff1) != 0:
                    x_inter = sd1 - diff1 * (sd2 - sd1) / (diff2 - diff1)
                    y_inter = f_capacity_interp_func(x_inter) 
                else:
                    st.warning(f"Iteration {trial+1}: Denominator is zero when calculating intersection, skipping this intersection search.")
                    x_inter = dpi 
                    y_inter = api
            else:
                st.warning(f"Iteration {trial+1}: No intersection found between capacity and current demand spectrum.")
                x_inter = dpi 
                y_inter = api

            if dpi == 0 or api == 0 or x_inter is None or y_inter is None:
                st.warning(f"Iteration {trial+1}: dpi, api, x_inter, or y_inter are invalid, cannot calculate error.")
                error = float('inf') 
            elif dpi <= 0 or api <= 0: 
                st.warning(f"Iteration {trial+1}: dpi or api are non-positive, error calculation might be inaccurate.")
                error = float('inf')
            else:
                error = (((x_inter - dpi)/dpi)**2 + ((y_inter - api)/api)**2)**0.5 * 100
            
            if error < 5:
                ax.plot(dpi, api, 'ro', markersize=8, label=f'Performance Point ({dpi:.3f}m, {api:.3f}g)')
                ax.plot([0, dy, dpi], [0, ay, api], linestyle='--', color='black', label='Bilinear Capacity Curve')
                st.success(f"✅ Performance Point found! Converged after {trial+1} iterations.")
                final_performance_point = (dpi, api, dmp_eff, trial + 1)
                break
            
            if x_inter is not None and not np.isnan(x_inter):
                dpi = max(dpi - 0.005, x_inter)
            else:
                st.warning(f"Iteration {trial+1}: Intersection displacement calculation is invalid, performance point might not converge.")
                dpi -= 0.005 
                if dpi < 0:
                    dpi = 0.01 
                    st.error("❌ Performance point displacement became negative, iteration stopped. Please check input data.")
                    break

            if trial == max_iterations - 1:
                st.warning(f"⚠️ Reached maximum iterations {max_iterations}, but performance point did not converge (final error: {error:.2f}%).")

        # --- 圖表設定與儲存到 session_state ---
        ax.set_xlabel('Spectral Displacement Sd (m)')
        ax.set_ylabel('Spectral Acceleration Sa (g)')
        ax.set_title("Capacity Spectrum & Demand Spectrum with Performance Point")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        max_x_limit = dpi * 3 
        max_y_limit = api * 5
        ax.set_xlim(left=0, right=max_x_limit) 
        ax.set_ylim(bottom=0, top=max_y_limit) 

        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.7), fontsize=8, ncol=1) 
        
        # 將生成的性能點圖存入 session_state
        st.session_state['fig_performance_point'] = fig 
        
        # 將性能點分析結果存入 session_state
        if final_performance_point:
            st.session_state['performance_point_results'] = {
                'final_dpi': final_performance_point[0],
                'final_api': final_performance_point[1],
                'final_dmp_eff': final_performance_point[2],
                'final_trial': final_performance_point[3]
            }
        elif not initial_dpi_found: 
            st.session_state['performance_point_results'] = "error" # 標記為錯誤狀態

    else:
        st.warning("⚠️ Please complete Step 3 (Capacity Spectrum Data) input or auto-conversion.")
        st.session_state['performance_point_results'] = None # 清除舊結果

    # 強制 Streamlit 重新運行，以便列中的圖表和結果能立即更新
    st.rerun() 


# -------------------------
# 結果顯示區塊 (統一顯示圖形與文字結果)
# -------------------------
st.markdown("---") 
st.header("📊 分析結果")

# 性能點分析結果文字顯示
if st.session_state['performance_point_results'] == "error":
    st.error("Failed to find performance point. Please check input data and parameters.")
elif st.session_state['performance_point_results'] is not None:
    results = st.session_state['performance_point_results']
    st.subheader("💡 性能點分析結果")
    st.markdown(f"**性能點位移 (Sd):** `{results['final_dpi']:.3f} m`")
    st.markdown(f"**性能點加速度 (Sa):** `{results['final_api']:.3f} g`")
    st.markdown(f"**有效阻尼比:** `{results['final_dmp_eff']:.2f}%`")
    st.markdown(f"**迭代次數:** `{results['final_trial']}`")
else:
    st.info("點擊 'Plot Capacity Spectrum & Performance Point' 按鈕來生成分析結果。")

# 創建兩列，讓圖表並排顯示
col_curve_plot, col_spectrum_plot = st.columns(2)

# 在左邊的列繪製容量曲線圖
with col_curve_plot:
    if df_curve is not None and not df_curve.empty:
        st.subheader("📉 容量曲線")
        # 繪製容量曲線並存入 session_state
        fig1, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax1.plot(df_curve['Displacement_m'], df_curve['BaseShear_tf'], marker='o', color='green', linewidth=2)
        ax1.set_xlabel("位移 (m)")
        ax1.set_ylabel("基底剪力 (tf)")
        ax1.set_title("容量曲線")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        st.session_state['fig_capacity_curve'] = fig1 # 存入 session_state
        st.pyplot(st.session_state['fig_capacity_curve'], use_container_width=True)
        get_image_download_link(st.session_state['fig_capacity_curve'], "capacity_curve.jpg", "⬇️ 下載容量曲線 (JPG)")
    else:
        st.info("⚠️ 請在 Step 2 貼上容量曲線資料以繪製此圖。")


# 在右邊的列繪製容量震譜與性能點圖
with col_spectrum_plot:
    if st.session_state['fig_performance_point'] is not None: 
        st.subheader("📈 容量震譜與性能點") 
        st.pyplot(st.session_state['fig_performance_point'], use_container_width=True)
        get_image_download_link(st.session_state['fig_performance_point'], "performance_point_spectrum.jpg", "⬇️ 下載容量/需求震譜 (JPG)")
    else:
        st.info("點擊上方的 'Plot Capacity Spectrum & Performance Point' 按鈕來生成此圖。")