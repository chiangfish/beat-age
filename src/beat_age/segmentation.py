import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import neurokit2 as nk

def segment(data, fs=360):
    """
    增强鲁棒性的多导联能量融合版 Pan-Tompkins + beat segmentation
    
    Parameters
    ----------
    data : np.ndarray, shape=(12, L)
        输入 ECG 信号，12 导联
    fs : int
        采样率，默认 360 Hz
    
    Returns
    -------
    rpeaks : list
        R 波峰索引列表（时间点，以采样点为单位）
    segments : list of np.ndarray
        每个元素是单个心搏，shape=(12, variable_length)，不重不漏覆盖全长
    """

    n_leads, L = data.shape

    # ----------------------------
    # 使用 NeuroKit 检测 R 峰
    # ----------------------------
    # 选择第一个导联作为主要导联进行 R 峰检测
    signals, info = nk.ecg_process(data[0], sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]

    # ----------------------------
    # 按 R-peak 分割心搏 (beat segmentation)
    # ----------------------------
    segments = []
    valid_rpeaks = []

    if len(rpeaks) > 0:
        # 以 R-R 间隔的中点作为分割点
        boundaries = [0]
        for i in range(len(rpeaks) - 1):
            midpoint = (rpeaks[i] + rpeaks[i+1]) // 2
            boundaries.append(midpoint)
        boundaries.append(L)

        for i in range(len(rpeaks)):
            start = boundaries[i]
            end = boundaries[i+1]
            
            if end > start:
                beat = data[:, start:end]
                segments.append(beat)
                valid_rpeaks.append(rpeaks[i])

    return valid_rpeaks, segments