ST_CSS = """
    <style>
    .custom-button {background-color: #4CAF50;color: white;padding: 10px 20px;font-size: 16px;border-radius: 5px;border: none;cursor: pointer;}
    .custom-button:hover {background-color: #45a049;}
    .stButton {display: flex;justify-content: center;}
    .stButton button {background-color: #55c9c2;color: white!important;padding: 12px 50px;font-size: 24px;border-radius: 5px;border: none;cursor: pointer;transition: all 0.2s ease;}
    .stButton button:hover {background-color: #4db5ae!important;color: white;}
    p.final_prediction {font-size: 42px;text-align: center;font-weight: 800;margin-top: 10px;}
    p.final_prediction.positive::after,p.final_prediction.negative::after {display: block;text-align:center;font-size: 40px;line-height: 50px;}
    p.final_prediction.positive::after {margin-top: 16px; content: "고객 유지";}
    p.final_prediction.negative::after {margin-top: 16px; content: "고객 이탈";}
    #root > div:nth-child(1) > div.withScreencast > div > div > section.stSidebar.st-emotion-cache-vmpjyt.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(5) > div > div > p {display: block; text-align: center; font-size: 16px; margin-bottom: 8px; }
    #root > div:nth-child(1) > div.withScreencast > div > div > section.stSidebar.st-emotion-cache-vmpjyt.eczjsme18 > div.st-emotion-cache-6qob1r.eczjsme11 > div.st-emotion-cache-1gwvy71.eczjsme12 > div > div > div > div > div:nth-child(6) > div > div > p {font-size: 13px; text-align: center;color: #999}
    </style>
    """
ST_TITLE = ('<h1 style="text-align: center; margin-bottom: 0">신용카드 이용 고객 이탈 예측</h1>')
ST_HEADER = ('<hr style="margin: 16px 0 36px; border-color: #ccc; border:2px solid #ddd"/>')
ST_SIDE_HEADER = ('<h3 style="margin-bottom: 0; transform: translateY(10px)">⚙️ 머신 러닝 모델</h2>')
ST_POSITIVE = '<p class="final_prediction positive" style="font-size: 70px;">🫶</p>'
ST_NEGATIVE = '<p class="final_prediction negative" style="font-size: 70px;">😭</p>'
