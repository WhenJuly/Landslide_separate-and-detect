import numpy as np
from matplotlib import pyplot as plt
import time as timing
import obspy
from utilities import mkdir
from torch_Tools import WaveformDataset, try_gpu
from obspy import Stream, Trace
import torch
from torch.utils.data import DataLoader
import matplotlib
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, decimate, spectrogram
from matplotlib.gridspec import GridSpec
matplotlib.rcParams.update({'font.size': 10})
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
time_start_time = timing.time()  # time when code starts

def plot_time_frequency(ax, data, dt, cmap='RdBu', reverse_cmap=False, xlim=None, ylim=None, vmin=None, vmax=None):
    EPSILON = 1e-12  # 微小偏移值
    f, t, Sxx = spectrogram(data, fs=1/dt, nperseg=128, noverlap=100)
    if reverse_cmap:
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + EPSILON), shading='gouraud', cmap=cmap+'_r', vmin=vmin, vmax=vmax)
    else:
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + EPSILON), shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frequency (Hz)', fontsize=14, fontfamily='Arial')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    return im

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
f_downsample = 4.9
def process_trace(tr, lowcut=0.5, highcut=4.9, f_downsample = f_downsample):
    tr.data = butter_bandpass_filter(tr.data, lowcut, highcut, tr.stats.sampling_rate)
    tr.data = decimate(tr.data, int(tr.stats.sampling_rate / f_downsample), zero_phase=False)
    tr.stats.sampling_rate = f_downsample
    data_mean = np.mean(tr.data)
    data_std = np.std(tr.data)
    tr.data = (tr.data - data_mean) / (data_std + 1e-12)
    tr.data = tr.data.astype(np.float32)
    return tr

# working_dir = 'G:/signal_separation/data_separation2'
# waveform_dir = r'I:\landquake\all_landslide_mseed\USA_mineland'
# waveform_mseed = waveform_dir + '/' + 'Z7.MOFO..HH_merge1.mseed'
waveform_dir = r'I:\landquake\all_landslide_mseed\190_BarryArm_09Aug2021'
waveform_mseed = waveform_dir + '/' + 'AK.BAE..BH_merge.mseed'
# output_file = waveform_dir + '/' + '253000512.2023.09_28_merge_after.mseed'
tr = obspy.read(waveform_mseed)
tr.merge(fill_value=0)

time_load_trace_time = timing.time() - time_start_time

for trace in tr:
    trace = process_trace(trace)

npts0 = tr[0].stats.npts
dt0 = tr[0].stats.delta
waveform0 = np.zeros((npts0, 3))
for i in range(3):
    waveform0[:, i] = tr[i].data

time0 = np.arange(0, npts0) * dt0

waveform_normalized = waveform0
print(waveform_normalized.shape)

length = waveform_normalized.shape[0]
new_length = (length + 600 - 1) // 600 * 600
padded_waveform = np.zeros((new_length, waveform_normalized.shape[1]))
print(padded_waveform.shape)
padded_waveform[:waveform_normalized.shape[0], :] = waveform_normalized
waveform_normalized = np.reshape(padded_waveform[:, np.newaxis, :], (-1, 600, 3))
print(waveform_normalized.shape)

# Predict the separated waveforms
waveform_data = WaveformDataset(waveform_normalized, waveform_normalized)

time_process_trace_time = timing.time() - time_start_time - time_load_trace_time

# -------------------------------------------------------------------------------
#|                         This part is about separation                        |
# -------------------------------------------------------------------------------

bottleneck_name1 = "ResidualMS_CAM_LSTM"  # ResidualMS_CAM_Transformer ResidualMS_CAM_LSTM
model_dataset_dir1 = r"I:\landslide_sep_det_loc_V\2separation\model2"
model_name1 = "Branch_Encoder_Decoder_" + bottleneck_name1
model_dir1 = model_dataset_dir1 + f'/{model_name1}' + '_warmup'

# Load model
model1 = torch.load(model_dir1 + '/' + f'{model_name1}_Model.pth', map_location='cpu')

batch_size = 64
test_iter1 = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)

# Test on real data
all_output_sep1 = np.zeros(waveform_normalized.shape)  # signal
all_output_sep2 = np.zeros(waveform_normalized.shape)  # noise
model1.eval()
for i, (X, _) in enumerate(test_iter1):
    print('+' * 12 + f'batch {i}' + '+' * 12)
    # output1, output2, feature_map1, feature_map2 = model(X)
    output_sep1 = model1(X)
    # output1 corresponds to earthquake signal
    output_sep1 = output_sep1.detach().numpy()
    output_sep1 = np.moveaxis(output_sep1, 1, -1)
    all_output_sep1[(i * batch_size): ((i + 1) * batch_size), :, :] = output_sep1

    # # output2 corresponds to ambient noise
    # output_sep2 = output_sep2.detach().numpy()
    # output_sep2 = np.moveaxis(output_sep2, 1, -1)
    # all_output_sep2[(i * batch_size): ((i + 1) * batch_size), :, :] = output_sep2

# Check the waveform
waveform_recovered = all_output_sep1
waveform_recovered = np.reshape(waveform_recovered, (-1, 3))
# # Save waveform_recovered as a numpy array
# np.save(r'G:\signal_separation\Data\滑坡数据5\CH.FUORN..HH_merge.npy', waveform_recovered)
# print('waveform_recovered saved as waveform_recovered.npy')

# noise_recovered = all_output_sep2
# noise_recovered = np.reshape(noise_recovered, (-1, 3))

waveform_original = np.reshape(waveform0, (-1, 3))
waveform_time = np.arange(padded_waveform.shape[0]) / 10

time_decompose_waveform = timing.time() - time_start_time - time_load_trace_time - time_process_trace_time

print('Time spent on decomposing seismograms: ')
print(f'Load mseed data: {time_load_trace_time:.3f} sec\n' +
      f'Process data (filtering, downsample, normalization): {time_process_trace_time:.3f} sec\n' +
      f'Decompose into earthquake and noise: {time_decompose_waveform:.3f} sec\n')

# -------------------------------------------------------------------------------
#|                         This part is about detection                        |
# -------------------------------------------------------------------------------

bottleneck_name2 = "ResidualMS_CAM_LSTM" #attention_LSTM ‘ResidualMS_CAM_LSTM’\  "ResidualMS_CAM_Transformer", "Linear", "LSTM", "ResidualMS_CAM_Transformer","MS_CAM_LSTM"
# model_dataset_dir = r"I:\landquake\all_landslide_mseed\processed_data\merge_landslide_earthquake\landslide_data\DATASET2\test1_normal_24360"
model_dataset_dir2 = r"I:\landslide_sep_det_loc_V\1detection\model\one_branch_2500noise_64"
# model_dataset_dir = r"I:\landquake\all_landslide_mseed\processed_data\merge_landslide_earthquake_noise\test1_DWA"
model_name2 = "Detection_one_branch_" + bottleneck_name2
model_dir2 = model_dataset_dir2 + f'/{model_name2}' + '_warmup'

model2 = torch.load(model_dir2 + '/' + f'{model_name2}_Model.pth', map_location='cpu')

batch_size = 64
test_iter2 = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)

# all_output1 = np.zeros(waveform_normalized.shape)
# all_output2 = np.zeros(waveform_normalized.shape)
all_output_det = np.zeros((waveform_normalized.shape[0], waveform_normalized.shape[1], 2))
model2.eval()
for i, (X, _) in enumerate(test_iter2):
    print('+' * 12 + f'batch {i}' + '+' * 12)
    output_det = model2(X)
    output_det = output_det.detach().numpy()
    output_det = np.moveaxis(output_det, 1, -1)
    all_output_det[(i * batch_size): ((i + 1) * batch_size), :, :] = output_det #按批次输出结果

waveform_detected = all_output_det
waveform_detected = waveform_detected.reshape(-1, 2)
# waveform_original = np.reshape(waveform0, (-1, 3))
# waveform_time = np.arange(padded_waveform.shape[0]) / f_downsample

# time_decompose_waveform = timing.time() - time_start_time - time_load_trace_time - time_process_trace_time

print('Time spent on decomposing seismograms: ')
print(f'Load mseed data: {time_load_trace_time:.3f} sec\n' +
      f'Process data (filtering, downsample, normalization): {time_process_trace_time:.3f} sec\n' +
      f'Detecte landslide: {time_decompose_waveform:.3f} sec\n')

# ylim=(-13, 13)
colors = [(r / 255, g / 255, b / 255) for r, g, b in [(254,254,254), (171,218,219), (254,35,34)]]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
# N = -700
vmin= -40
vmax= 20
# print(waveform_recovered.shape)
fig, axes = plt.subplots(7, 1, figsize=(8, 8))  # 7 rows, 1 column for subplots

for i in range(3):
    ax1 = axes[2*i]  # Waveform plot
    ax2 = axes[2*i + 1]  # Time-frequency spectrogram plot

    # Plotting the waveform
    ax1.plot(waveform_time[:len(waveform0[:, 0])], waveform_recovered[:len(waveform0[:, 0]), i], color='#2f4f4f', label=f'Component {i+1} Waveform')
    ax1.set_ylim(-10, 10)
    ax1.set_xlim(0, waveform0.shape[0]/f_downsample)
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Amplitude')

    # Plotting the time-frequency spectrogram
    # im = plot_time_frequency(ax2, waveform_recovered[:len(waveform0[:, 0]), i], dt0, xlim=(0, waveform0.shape[0]/f_downsample), ylim=(0.01, 5),vmin=vmin, vmax=vmax, cmap=cmap)
    # # ax2.set_yscale('log')
    # ax2.set_yticks([0.01, 0.1, 1, 5])
    # ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Frequency (Hz)')
#
# # 绘制识别结果，线条为玫红色
# print(waveform_time.shape)
# print(waveform0.shape)
# print(waveform_detected.shape)

ax2 = axes[-1]
ax2.plot(waveform_time[:len(waveform0[:, 0])], waveform_detected[:len(waveform0[:, 0]), 0], color='#791c3f', linestyle='-.', linewidth=2, label='Recognition Result')
# ax2.plot(waveform_time[:len(waveform0[:, 0])], waveform_detected[:len(waveform0[:, 0])], color='#791c3f', label='Recognition Result') #画两条线
# ax2.plot(waveform_time[:len(waveform_detected[:, 1])], waveform_detected[:, 1], color='#4682B4', label='Recognition Result')
# 添加y=1和y=2的灰色虚线
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax2.set_ylim(0, 1)
ax2.set_xlim(0, len(waveform0[:, 0])/f_downsample)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Recognition Output')
plt.tight_layout()
plt.show()

# 保存结果
# for k in range(3):
#     waveform0[:, k] = waveform_recovered[:len(waveform0[:, 0]), k]
#     with open(f'I:\landslide_sep_det_loc_V\\figure\\190_BarryArm_09Aug2021\\waveform_separate_{k}.txt', 'w') as f:
#         f.write('\n'.join(map(str, waveform0[:, k])))
#     print(f'Component {k} saved to waveform_component_{k}.txt')

# np.savetxt("G:\signal_separation\Figuer\Method comprision\original_signal.txt", waveform0[:,0], fmt="%.10f")
# np.savetxt("G:\signal_separation\Figuer\Method comprision\Bi-LSTM-with-residual.txt", waveform_recovered[:,0], fmt="%.10f")

# with open(r'I:\landslide_sep_det_loc_V\figure\色东普监测\Station\253000512\detection_0928.txt', 'w') as f:
#     f.write('\n'.join(map(str, waveform_detected[:len(waveform0[:, 0]), 0])))
# print(f' saved to detection_proposed.txt')

# 保存分离后的mseed文件
# # 获取三分量数据
# stream = Stream()
# for i in range(3):
#     trace_data = waveform_recovered[:len(waveform0[:, 0]), i]
#     trace = Trace(data=trace_data.astype('float32'))  # 确保数据类型适合MSEED格式
#
#     # 设置最简单的头信息（仅必要信息）
#     trace.stats.network = ""
#     trace.stats.station = ""
#     trace.stats.channel = ""
#     trace.stats.sampling_rate = 10  # 保证采样率正确
#
#     # 将 Trace 添加到 Stream
#     stream.append(trace)
#
# # 保存为 .mseed 文件
# stream.write(output_file, format="MSEED")








# # Normalize feature map
# feature_map_normalized = 2 * (feature_map1 - torch.min(feature_map1)) / (torch.max(feature_map1) - torch.min(feature_map1)) - 1
#
# # Convert to numpy for plotting
# feature_map_np = feature_map_normalized.detach().cpu().numpy()  # Move back to CPU for plotting
#
# # Reshape feature map to combine all channels (components) into one matrix
# combined_feature_map = feature_map_np[0].reshape(-1, feature_map_np.shape[-1])
# # Define custom colors for the colormap
# colors = [(r / 255, g / 255, b / 255) for r, g, b in [(10,100,100), (255,212,114), (179,13,0)]] # Red, Green, Blue
# # Create a LinearSegmentedColormap object
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
# # Plot the combined feature map
# fig, axs = plt.subplots(2, 1, figsize=(5, 5))
#
# # Plot waveform
# normalized_waveform0 = (waveform0[:,2] - np.mean(waveform0[:,2])) / (np.max(abs(waveform0[:,2])))
# plt.subplot(2, 1, 1)
# plt.plot(normalized_waveform0, color=(138/255, 136/255, 131/255))#138/255, 136/255, 131/255 noise   173/255, 60/255, 4/255 landslide  0/255, 45/255, 72/255 earthquake
# plt.ylabel('Amplitude', style='normal', size=18, family='Arial')
# plt.xticks([])
# plt.yticks(style='normal', size=14, family='Arial')
# # plt.title(f'Waveform')
# plt.xlim(0, len(waveform0))
# plt.ylim(-1, 1)
#
# plt.subplot(2, 1, 2)
# plt.imshow(combined_feature_map, aspect='auto', origin='lower', cmap=cmap) #coolwarm_r  Spectral_r  PuOr  RdBu_r  cubehelix_r pink_r 'YlOrBr'
# plt.xlabel('Timestamp', style='normal', size=18, family='Arial')
# plt.ylabel('Feature dims', style='normal', size=18, family='Arial')
# plt.xticks(np.linspace(0, combined_feature_map.shape[1] - 1, 3), np.linspace(0, 600, 3), style='normal', size=14, family='Arial')
# plt.yticks(np.linspace(0, 64, 3), style='normal', size=14, family='Arial')
# plt.colorbar(label='')
# plt.xlim(0, 74)
# # plt.title(f'Combined Feature Map Heatmap - {waveform_mseed}')
#
# plt.tight_layout()
# plt.show()
# # plt.savefig(r"G:\signal_separation\Figuer\featuremap_colormap.jpg", dpi=600, format="jpg")


##
# import numpy as np
# from matplotlib import pyplot as plt
# import time as timing
# import obspy
# import pywt
# from utilities import mkdir
# from torch_tools import WaveformDataset, try_gpu
# import torch
# from torch.utils.data import DataLoader
# import matplotlib
# from scipy.signal import butter, filtfilt, decimate, spectrogram
#
# matplotlib.rcParams.update({'font.size': 10})
#
# time_start_time = timing.time()  # time when code starts
#
# def plot_time_frequency(ax, data, dt, cmap='RdBu', reverse_cmap=True, xlim=None, ylim=None, vmin=None, vmax=None):
#     EPSILON = 1e-12  # 微小偏移值
#     f, t, Sxx = spectrogram(data, fs=1/dt, nperseg=128, noverlap=120)
#     if reverse_cmap:
#         im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + EPSILON), shading='gouraud', cmap=cmap+'_r', vmin=vmin, vmax=vmax)
#     else:
#         im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + EPSILON), shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.set_ylabel('Frequency (Hz)')
#     if xlim:
#         ax.set_xlim(xlim)
#     if ylim:
#         ax.set_ylim(ylim)
#     return im
#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data)
#     return y
#
# def process_trace(tr, lowcut=0.5, highcut=5, f_downsample=10): #0.5 5
#     # 滤波
#     tr.data = butter_bandpass_filter(tr.data, lowcut, highcut, tr.stats.sampling_rate)
#     # 降采样
#     tr.data = decimate(tr.data, int(tr.stats.sampling_rate / f_downsample), zero_phase=False)
#     tr.stats.sampling_rate = f_downsample
#     return tr
#
# def normalize_window(window):
#     # 对每个通道进行标准化
#     mean = np.mean(window, axis=0)
#     std = np.std(window, axis=0) + 1e-12
#     normalized_window = (window - mean) / std
#     return normalized_window, mean, std
#
# def denormalize_window(window, mean, std):
#     return window * std + mean
#
# def pad_waveform(waveform, window_size=600):
#     length = waveform.shape[0]
#     new_length = (length + window_size - 1) // window_size * window_size  # 向上取整到最近的window_size倍数
#     padded_waveform = np.zeros((new_length, waveform.shape[1]))
#     padded_waveform[:waveform.shape[0], :] = waveform
#     return padded_waveform
#
# working_dir = 'G:/signal_separation/data_separation2'
# waveform_dir = r'I:\landquake\地震局滑坡数据\新磨滑坡\2017062405\mseed'
# waveform_mseed = waveform_dir + '/' + 'MXI.00.BH_merge.mseed'
#
# tr = obspy.read(waveform_mseed)
# tr.merge(fill_value=0)  # in case that there are segmented traces
#
# time_load_trace_time = timing.time() - time_start_time  # time spent on loading data
#
# # Apply preprocessing to each trace
# for trace in tr:
#     trace = process_trace(trace)
#
# # Reformat the waveform data into array
# npts0 = tr[0].stats.npts  # number of samples
# dt0 = tr[0].stats.delta  # dt
# waveform0 = np.zeros((npts0, 3))
# for i in range(3):
#     waveform0[:, i] = tr[i].data
#
# time0 = np.arange(0, npts0) * dt0
#
# # Pad the waveform to the nearest multiple of 600
# padded_waveform = pad_waveform(waveform0, window_size=600)
#
# # Split the waveform into windows of size 600
# windows = np.reshape(padded_waveform, (-1, 600, 3))
#
# # Normalize each window
# normalized_windows = []
# means = []
# stds = []
# for window in windows:
#     normalized_window, mean, std = normalize_window(window)
#     normalized_windows.append(normalized_window)
#     means.append(mean)
#     stds.append(std)
#
# normalized_windows = np.array(normalized_windows)
# means = np.array(means)
# stds = np.array(stds)
#
# # Reformat the data into the format required by the model (batch, channel, samples)
# waveform_normalized = np.transpose(normalized_windows, (0, 2, 1))
# print(waveform_normalized.shape)
#
# # Predict the separated waveforms
# waveform_data = WaveformDataset(waveform_normalized, waveform_normalized)
#
# # time spent on preprocessing data
# time_process_trace_time = timing.time() - time_start_time - time_load_trace_time
#
# # Specify model name
# bottleneck_name = "ResidualMS_CAM_LSTM"  # ResidualMS_CAM_Transformer
# model_dataset_dir = "G:\signal_separation\data_separation2\continue seismic recordings\Model_test_18061_GPU_SNR1-8-MSE-PEARSON1_1"
# model_name = "Branch_Encoder_Decoder_" + bottleneck_name
# model_dir = model_dataset_dir + f'/{model_name}' + '_warmup'
#
# # Load model
# model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location='cpu')
#
# batch_size = 128
# test_iter = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)
#
# # Test on real data
# all_output1 = np.zeros(normalized_windows.shape)  # signal
# all_output2 = np.zeros(normalized_windows.shape)  # noise
# model.eval()
# for i, (X, _) in enumerate(test_iter):
#     print('+' * 12 + f'batch {i}' + '+' * 12)
#     X = X.permute(0, 2, 1)  # 将 X 从 [batch_size, samples, channels] 转换为 [batch_size, channels, samples]
#     output1, output2 = model(X)
#
#     # output1 corresponds to earthquake signal
#     output1 = output1.detach().numpy()
#     output1 = np.moveaxis(output1, 1, -1)
#     all_output1[(i * batch_size): ((i + 1) * batch_size), :, :] = output1
#
#     # output2 corresponds to ambient noise
#     output2 = output2.detach().numpy()
#     output2 = np.moveaxis(output2, 1, -1)
#     all_output2[(i * batch_size): ((i + 1) * batch_size), :, :] = output2
#
# # Denormalize the windows
# denormalized_waveform_recovered = []
# denormalized_noise_recovered = []
# for i in range(len(normalized_windows)):
#     denormalized_waveform_recovered.append(denormalize_window(all_output1[i], means[i], stds[i]))
#     denormalized_noise_recovered.append(denormalize_window(all_output2[i], means[i], stds[i]))
#
# waveform_recovered = np.concatenate(denormalized_waveform_recovered, axis=0)
# noise_recovered = np.concatenate(denormalized_noise_recovered, axis=0)
#
# waveform_original = np.reshape(waveform0, (-1, 3))
# waveform_time = np.arange(padded_waveform.shape[0]) / 10
#
# time_decompose_waveform = timing.time() - time_start_time - time_load_trace_time - time_process_trace_time
#
# print('Time spent on decomposing seismograms: ')
# print(f'Load one-month mseed data: {time_load_trace_time:.3f} sec\n' +
#       f'Process data (filtering, downsample, normalization): {time_process_trace_time:.3f} sec\n' +
#       f'Decompose into earthquake and noise: {time_decompose_waveform:.3f} sec\n')
#
# fig, axs = plt.subplots(3, 3, figsize=(12, 8))
# titles = ['Channel E', 'Channel N', 'Channel Z']
#
# N = -700
# # 绘制第一列：waveform_mseed
# for i in range(3):
#     axs[i, 0].plot(time0[:N], waveform0[:N, i], color='#708090')
#     axs[i, 0].set_title(titles[i] + ' (Original)')
#
# # 绘制第二列：waveform_recovered
# for i in range(3):
#     axs[i, 1].plot(waveform_time[:N], waveform_recovered[:N, i], color='#FF6A6A')
#     axs[i, 1].set_title(titles[i] + ' (landslide)')
#
# # 绘制第三列：noise_recovered
# for i in range(3):
#     axs[i, 2].plot(waveform_time[:N], noise_recovered[:N, i], color='#4682B4')
#     axs[i, 2].set_title(titles[i] + ' (Noise)')
# plt.show()
#
# fig, axs = plt.subplots(6, 3, figsize=(12, 8))
# titles = ['Channel E', 'Channel N', 'Channel Z']
#
# # 绘制第一列：waveform_mseed
# N = -700
# for i in range(3):
#     waveform_norm = (waveform0[:N, i] - np.mean(waveform0[:N, i])) / np.std(waveform0[:N, i])
#     axs[2*i, 0].plot(time0[:N], waveform_norm, color='#708090')
#     axs[2*i, 0].set_title(titles[i] + ' Original')
#     axs[2*i+1, 0].set_title('Time-Frequency Spectrum')
#     im = plot_time_frequency(axs[2*i + 1, 0], waveform0[:N, i], dt0, ylim=(0, 5), vmin=-30, vmax=30)
#
# # 绘制第二列：waveform_recovered
# for i in range(3):
#     waveform_norm = (waveform_recovered[:N, i] - np.mean(waveform_recovered[:N, i])) / np.std(waveform_recovered[:N, i])
#     axs[2*i, 1].plot(waveform_time[:N], waveform_norm, color='#FF6A6A')
#     axs[2*i, 1].set_title(titles[i] + ' Landslide')
#     axs[2*i+1, 1].set_title('Time-Frequency Spectrum')
#     im = plot_time_frequency(axs[2*i + 1, 1], waveform_recovered[:N, i], dt0, ylim=(0, 5), vmin=-30, vmax=30)
#
# # 绘制第三列：noise_recovered
# for i in range(3):
#     waveform_norm = (noise_recovered[:N, i] - np.mean(noise_recovered[:N, i])) / np.std(noise_recovered[:N, i])
#     axs[2*i, 2].plot(waveform_time[:N], waveform_norm, color='#4682B4')
#     axs[2*i, 2].set_title(titles[i] + ' Noise')
#     axs[2*i+1, 2].set_title('Time-Frequency Spectrum')
#     im = plot_time_frequency(axs[2*i + 1, 2], noise_recovered[:N, i], dt0, ylim=(0, 5), vmin=-30, vmax=30)
# plt.show()
