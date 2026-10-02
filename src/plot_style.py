"""노트북 그림 공통 설정 : 한글 폰트를 OS 에 맞게 찾는다.

macOS · Windows · Linux 순서로 알려진 경로를 보고, 없으면 설치된 글꼴 이름에서 찾는다.
그래도 없으면 경고만 내고 기본 글꼴로 계속 그린다 (그림의 한글이 네모로 보일 수 있다).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT_FILES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",          # macOS
    "C:/Windows/Fonts/malgun.ttf",                          # Windows (맑은 고딕)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Ubuntu·Debian (fonts-nanum)
    "/usr/share/fonts/nanum/NanumGothic.ttf",               # Fedora 등
]
FONT_NAMES = ["Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic",
              "Noto Sans CJK KR", "Noto Sans KR"]


def set_korean_font():
    """한글 폰트를 rcParams 에 설정하고 고른 글꼴 이름을 돌려준다 (못 찾으면 None)."""
    name = None
    for path in FONT_FILES:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            break
    if name is None:
        installed = {f.name for f in fm.fontManager.ttflist}
        name = next((n for n in FONT_NAMES if n in installed), None)
    if name:
        plt.rcParams["font.family"] = name
    else:
        warnings.warn("한글 폰트를 찾지 못해 기본 폰트로 그린다. 그림의 한글이 네모로 보일 수 있다.")
    plt.rcParams["axes.unicode_minus"] = False
    return name
