# app.py ーーー Google Sheets版：登録/入力/ダッシュボード/設定（スマホ最適）
import streamlit as st, pandas as pd, altair as alt, datetime as dt
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials

# ===== ページ設定（スマホ寄せ） =====
st.set_page_config(page_title="健康管理（Sheets版）", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>.block-container{padding-top:.6rem;padding-bottom:.6rem}.stMetric{text-align:center}</style>
""", unsafe_allow_html=True)

# ===== 認証（サービスアカウント）=====
SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open_by_url(st.secrets["spreadsheet_url"])
ws_users   = sh.worksheet("USERS")
ws_entries = sh.worksheet("ENTRIES")

# ===== 便利関数 =====
def load_users(active_only=True)->pd.DataFrame:
    df = get_as_dataframe(ws_users, evaluate_formulas=True, header=0).dropna(how="all")
    if df.empty: 
        return pd.DataFrame(columns=["NAME","HEIGHT_CM","TARGET_WEIGHT","TARGET_BODYFAT","TARGET_MUSCLE","TARGET_BMI","IS_ACTIVE","CREATED_AT","UPDATED_AT"])
    # 型調整
    for c in ["HEIGHT_CM","TARGET_WEIGHT","TARGET_BODYFAT","TARGET_MUSCLE","TARGET_BMI"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "IS_ACTIVE" in df:
        df["IS_ACTIVE"] = df["IS_ACTIVE"].astype(str).str.lower().isin(["true","1","yes"])
    else:
        df["IS_ACTIVE"] = True
    if active_only: df = df[df["IS_ACTIVE"]==True]
    return df.sort_values("NAME")

def save_users(df:pd.DataFrame):
    # 列順を固定
    cols = ["NAME","HEIGHT_CM","TARGET_WEIGHT","TARGET_BODYFAT","TARGET_MUSCLE","TARGET_BMI","IS_ACTIVE","CREATED_AT","UPDATED_AT"]
    for c in cols:
        if c not in df.columns: df[c] = None
    set_with_dataframe(ws_users, df[cols], include_index=False, include_column_header=True, resize=True)

def load_entries()->pd.DataFrame:
    df = get_as_dataframe(ws_entries, evaluate_formulas=True, header=0).dropna(how="all")
    if df.empty:
        return pd.DataFrame(columns=["ENTRY_DATE","NAME","WEIGHT","BODYFAT","MUSCLE","MEMO","CREATED_AT"])
    df["ENTRY_DATE"] = pd.to_datetime(df["ENTRY_DATE"], errors="coerce").dt.date
    for c in ["WEIGHT","BODYFAT","MUSCLE"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.sort_values(["ENTRY_DATE","NAME"])

def append_entry(row:dict):
    vals = [row.get(k) for k in ["ENTRY_DATE","NAME","WEIGHT","BODYFAT","MUSCLE","MEMO","CREATED_AT"]]
    ws_entries.append_row(vals, value_input_option="USER_ENTERED")

def bmi(weight, height_cm):
    if not weight or not height_cm or height_cm<=0: return None
    h = height_cm/100
    return round(weight/(h*h), 2)

# モバイルレイアウト
with st.sidebar:
    st.markdown("### 表示設定")
    IS_MOBILE = st.toggle("モバイル簡易モード", value=True)
def cols(n_desktop, n_mobile=1):
    return st.columns(n_mobile if IS_MOBILE else n_desktop)
CHART_H = 240 if IS_MOBILE else 280
DIFF_H  = 200 if IS_MOBILE else 220

# 換算パラメータ
with st.sidebar:
    st.markdown("### 変換パラメータ")
    kcal_per_kg = st.number_input("1kg減に必要kcal", 1000, 15000, 7700, 100)
    onigiri_kcal= st.number_input("おにぎり1個のkcal", 50, 500, 180, 10)
    run_coef    = st.number_input("ラン係数(kcal/[kg・km])", 0.5, 1.5, 1.0, 0.05)
    passcode    = st.text_input("パスコード(任意)", type="password", placeholder="空なら無効")

# 任意の簡易パスコード（必要ならSecretsに設定し、ここで比較）
if st.secrets.get("app_passcode"):
    if passcode != st.secrets["app_passcode"]:
        st.warning("パスコードを入力してください（管理者に確認）")
        st.stop()

# ===== ページ選択 =====
page = st.sidebar.radio("ページ", ["登録（ユーザー追加）","入力","ダッシュボード（全員）","設定（目標）"])

# ===== ページ：登録 =====
if page=="登録（ユーザー追加）":
    st.subheader("新規ユーザー登録 / 更新")
    udf = load_users(active_only=False)

    with st.form("reg"):
        name = st.text_input("名前（必須）").strip()
        c1,c2 = st.columns(2)
        with c1:
            height = st.number_input("身長(cm)", 50.0, 250.0, step=0.1)
            t_w    = st.number_input("目標体重(kg)", 0.0, 300.0, step=0.1)
            t_bmi  = st.number_input("目標BMI", 0.0, 60.0, step=0.1)
        with c2:
            t_bf   = st.number_input("目標体脂肪率(%)", 0.0, 100.0, step=0.1)
            t_ms   = st.number_input("目標骨格筋率(%)", 0.0, 100.0, step=0.1)
            active = st.checkbox("有効", value=True)
        ok = st.form_submit_button("登録 / 更新", type="primary")

    if ok:
        if not name:
            st.error("名前は必須です")
        else:
            now = dt.datetime.now()
            if (udf["NAME"]==name).any():
                udf.loc[udf["NAME"]==name, ["HEIGHT_CM","TARGET_WEIGHT","TARGET_BODYFAT","TARGET_MUSCLE","TARGET_BMI","IS_ACTIVE","UPDATED_AT"]] = \
                    [height,t_w,t_bf,t_ms,t_bmi,active,now]
            else:
                new = pd.DataFrame([{
                    "NAME":name, "HEIGHT_CM":height, "TARGET_WEIGHT":t_w, "TARGET_BODYFAT":t_bf,
                    "TARGET_MUSCLE":t_ms, "TARGET_BMI":t_bmi, "IS_ACTIVE":active, "CREATED_AT":now, "UPDATED_AT":now
                }])
                udf = pd.concat([udf, new], ignore_index=True)
            save_users(udf)
            st.success(f"『{name}』を登録/更新しました。")

# ===== ページ：入力 =====
elif page=="入力":
    st.subheader("毎日の記録を入力")
    users = load_users()
    options = users["NAME"].tolist() + ["＋新規登録…"] if not users.empty else ["＋新規登録…"]
    sel = st.selectbox("名前", options=options)
    if sel=="＋新規登録…":
        st.info("『登録』ページでユーザーを追加してください。")
        st.stop()
    name = sel

    # 前回値
    edf = load_entries()
    prev = edf[edf["NAME"]==name].sort_values("ENTRY_DATE").tail(1)
    box = st.container(border=True)
    if not prev.empty:
        p = prev.iloc[0]
        c1,c2,c3,c4 = cols(4,2)
        c1.write(f"体重: {p['WEIGHT']:.1f} kg")
        c2.write(f"体脂肪率: {p['BODYFAT']:.1f} %")
        c3.write(f"骨格筋率: {p['MUSCLE']:.1f} %")
        c4.write(f"日付: {p['ENTRY_DATE']}")

    c1,c2 = st.columns(2)
    with c1:
        entry_date = st.date_input("日付", dt.date.today())
        weight = st.number_input("体重(kg)", 0.0, 300.0, step=0.1, format="%.1f")
    with c2:
        bf = st.number_input("体脂肪率(%)", 0.0, 100.0, step=0.1, format="%.1f")
        ms = st.number_input("骨格筋率(%)", 0.0, 100.0, step=0.1, format="%.1f")
    memo = st.text_input("メモ（任意）")

    # 異常値チェック
    need_confirm = False
    if not prev.empty and weight:
        dw = float(weight) - float(prev.iloc[0]["WEIGHT"])
        if abs(dw)>=3.0:
            st.warning(f"前回比 {dw:+.1f} kg。誤入力かも？")
            need_confirm = True
    ok_flag = True if not need_confirm else st.checkbox("確認しました（この数値で保存）")

    if st.button("保存", type="primary", disabled=need_confirm and not ok_flag):
        append_entry({
            "ENTRY_DATE":entry_date, "NAME":name, "WEIGHT":weight, "BODYFAT":bf, "MUSCLE":ms,
            "MEMO":memo, "CREATED_AT":dt.datetime.now()
        })
        st.success("保存しました ✅")

# ===== ページ：ダッシュボード =====
elif page=="ダッシュボード（全員）":
    st.subheader("全員の推移と最新状況")
    udf = load_users()
    edf = load_entries()
    df = edf.merge(udf, on="NAME", how="inner")
    if df.empty:
        st.info("データがありません")
        st.stop()

    # フィルタ（本文側）
    f1,f2 = cols(2,1)
    with f1:
        user_opt = ["（全員）"] + sorted(df["NAME"].unique().tolist())
        sel_u = st.selectbox("ユーザー", user_opt, index=0)
    with f2:
        period = st.selectbox("期間", ["全期間","直近30日","直近90日"], index=1)

    # 期間フィルタ & BMI
    dff = df.copy()
    dff["ENTRY_DATE"] = pd.to_datetime(dff["ENTRY_DATE"])
    if period!="全期間":
        days = 30 if period=="直近30日" else 90
        dff = dff[dff["ENTRY_DATE"] >= (pd.Timestamp.today().normalize() - pd.Timedelta(days=days))]
    dff["BMI"] = dff.apply(lambda r: bmi(r["WEIGHT"], r["HEIGHT_CM"]), axis=1)
    if sel_u!="（全員）":
        dff = dff[dff["NAME"]==sel_u]

    # 最新値カード + アクション量
    st.markdown("### 最新値（人別）")
    for n, g in dff.sort_values("ENTRY_DATE").groupby("NAME"):
        last = g.iloc[-1]
        k1,k2,k3,k4 = cols(4,2)
        k1.metric(f"{n} 体重", f"{last['WEIGHT']:.1f} kg")
        k2.metric("体脂肪率", f"{last['BODYFAT']:.1f} %")
        k3.metric("骨格筋率", f"{last['MUSCLE']:.1f} %")
        k4.metric("BMI", f"{last['BMI']:.1f}")

        need_kcal = max(0.0, (last["WEIGHT"] - float(last.get("TARGET_WEIGHT") or 0)))*kcal_per_kg if pd.notna(last.get("TARGET_WEIGHT")) else None
        if need_kcal is not None:
            kg_diff = max(0.0, last["WEIGHT"] - float(last["TARGET_WEIGHT"]))
            oni = (need_kcal/onigiri_kcal) if onigiri_kcal>0 else None
            dist = (need_kcal/(last["WEIGHT"]*run_coef)) if last["WEIGHT"] and run_coef>0 else None
            c1,c2,c3 = cols(3,1)
            c1.metric("必要消費kcal", f"{need_kcal:,.0f} kcal", f"{kg_diff:.1f}kg")
            if oni is not None:  c2.metric("おにぎり換算", f"{oni:.1f} 個")
            if dist is not None: c3.metric("必要ラン距離", f"{dist:.1f} km")
        else:
            st.caption("目標体重が未設定です")

    # 折れ線（目標ライン付）
    def line_with_goal(df_plot, y, title, goal_col):
        base = alt.Chart(df_plot).mark_line(point=not IS_MOBILE).encode(
            x="ENTRY_DATE:T", y=alt.Y(f"{y}:Q", title=title), color="NAME:N"
        )
        layers=[base]
        for n, sub in df_plot.groupby("NAME"):
            gv = sub[goal_col].dropna()
            if len(gv):
                goal = float(gv.iloc[-1])
                layers.append(alt.Chart(sub).mark_rule(strokeDash=[4,4]).encode(y=alt.datum(goal)))
                layers.append(alt.Chart(sub.iloc[-1:]).mark_text(dy=-8).encode(
                    x="ENTRY_DATE:T", y=alt.datum(goal), text=alt.value(f"{n} 目標")
                ))
        st.altair_chart(alt.layer(*layers).properties(height=CHART_H), use_container_width=True)

    st.markdown("### 推移グラフ（目標ライン付）")
    line_with_goal(dff, "WEIGHT", "体重(kg)", "TARGET_WEIGHT")
    line_with_goal(dff, "BODYFAT", "体脂肪率(%)", "TARGET_BODYFAT")
    line_with_goal(dff, "MUSCLE",  "骨格筋率(%)", "TARGET_MUSCLE")
    line_with_goal(dff, "BMI",     "BMI",        "TARGET_BMI")

    # 乖離
    st.markdown("### 目標との差分（体重）")
    dd = dff.copy()
    dd["WEIGHT_DIFF"] = dd["WEIGHT"] - dd["TARGET_WEIGHT"]
    diff = alt.Chart(dd).mark_line(point=not IS_MOBILE).encode(
        x="ENTRY_DATE:T", y=alt.Y("WEIGHT_DIFF:Q", title="体重差分(kg)"), color="NAME:N"
    )
    zero = alt.Chart(dd).mark_rule(strokeDash=[2,2]).encode(y=alt.datum(0))
    st.altair_chart((diff+zero).properties(height=DIFF_H), use_container_width=True)

# ===== ページ：設定 =====
else:
    st.subheader("ユーザー設定（身長・目標・有効化）")
    df = load_users(active_only=False)
    if df.empty:
        st.info("ユーザーが未登録です。『登録』で作成してください。")
    else:
        edit = df.rename(columns={
            "NAME":"名前","HEIGHT_CM":"身長(cm)","TARGET_WEIGHT":"目標体重",
            "TARGET_BODYFAT":"目標体脂肪率","TARGET_MUSCLE":"目標骨格筋率","TARGET_BMI":"目標BMI",
            "IS_ACTIVE":"有効","CREATED_AT":"作成日時","UPDATED_AT":"更新日時"
        })
        edited = st.data_editor(edit, use_container_width=True, column_config={
            "作成日時": st.column_config.DatetimeColumn(disabled=True),
            "更新日時": st.column_config.DatetimeColumn(disabled=True),
        })
        if st.button("保存", type="primary"):
            save = edited.rename(columns={
                "名前":"NAME","身長(cm)":"HEIGHT_CM","目標体重":"TARGET_WEIGHT",
                "目標体脂肪率":"TARGET_BODYFAT","目標骨格筋率":"TARGET_MUSCLE",
                "目標BMI":"TARGET_BMI","有効":"IS_ACTIVE"
            })
            now = dt.datetime.now()
            if "UPDATED_AT" in save.columns: save["UPDATED_AT"]=now
            if "CREATED_AT" not in save.columns: save["CREATED_AT"]=now
            save_users(save)
            st.success("保存しました ✅")
