import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from scipy.fft import fft, fftfreq

def load_harmonic_data(hf, amplitudes, frequency, downsampling_factor):
    harmonic_data = {}
    for amp in amplitudes:
        key = f'Harm_A{amp}_f{frequency}_Q'
        data = np.array(hf.get(key))
        data = data[::downsampling_factor]
        harmonic_data[amp] = data

    return harmonic_data

def prepare_input_signal(frequency, duration, sample_length, downsampling_factor, timeHistorySizeOfU):
    t = np.linspace(0, duration, sample_length)
    u = np.sin(2 * np.pi * frequency * t)
    fake_velocity = np.zeros(timeHistorySizeOfU)
    u = np.concatenate((fake_velocity, u))

    u_reshaped = np.lib.stride_tricks.sliding_window_view(u, timeHistorySizeOfU)
    u_reshaped = u_reshaped[::downsampling_factor, :]
    u_reshaped = torch.tensor(u_reshaped, dtype=torch.float32)
    
    return u_reshaped

def plot_predictions(harmonic_data, predictions, model_name, path):
    for amp, data in harmonic_data.items():
        plt.figure(figsize=(12, 8))
        plt.plot(data, label='Harmonic Data', color='blue', alpha=0.5)
        plt.plot(predictions[amp], label='Prediction', color='red')
        plt.xlabel('Input Feature')
        plt.ylabel('Output')
        plt.title(f'A = {amp}%')
        plt.legend()
        plt.savefig(f'figures/{path}/harmonic_A{amp}.png')
        plt.close()

def compute_harmonics_error(pred, y, A):
    pred = pred.squeeze(-1)
    mse = np.sum((pred-y)**2) / len(pred)
    mse_relative = mse / np.mean(y ** 2)
    mse_harmonics = mse / (A ** 2)
    return mse, mse_relative, mse_harmonics

def Get_gain_phase(Q, A, f, dt=1e-4):
    Q_end = Q.shape[0]
    nn = Q.shape[0]
    
    # Calculate frequency resolution
    while nn > (Q_end/2):
        freq_res = 1 / (nn * dt)
        if (f / freq_res) % 1 < 1e-3:
            Q_end = nn
            break
        nn -= 1
    Q = Q[:Q_end]
    N = Q.shape[0]
    
    # Perform FFT and get frequencies
    yf = fft(Q, N)
    xf = fftfreq(N, dt)
    
    # Find the closest frequency to f
    idx = np.argmin(np.abs(xf - f))
    
    # Check if the frequency is within the allowed tolerance
    if np.abs(xf[idx] - f) <= 0.2:
        FDF = 2 * np.abs(yf[idx] / N) / A
        FDFphase = np.angle(yf[idx]) + np.pi / 2
        if FDFphase > np.pi:
            FDFphase -= 2 * np.pi
    else:
        FDF, FDFphase = 0, 0
    
    return FDF, FDFphase

def calculate_FDF(A,h, model, timeHistorySizeOfU, downsampling_factor):
    # A = Amplitude
    # h = harmonic number
 
    gain = np.zeros((50))
    phase = np.zeros((50))
    
    frequencies = np.zeros(50)
    jj = 0
    
    for freq in range(5, 500, 10):
        t = np.linspace(0, 0.5, 5000)
        u =  A * np.sin(2 * np.pi * freq * t)
        fake_velocity = np.zeros(timeHistorySizeOfU-1)
        u = np.concatenate((fake_velocity,u))
        #u = np.squeeze(input_scaler.transform(u.reshape(-1,1)))
        u_reshaped = np.lib.stride_tricks.sliding_window_view(u, timeHistorySizeOfU)
        u_reshaped = np.array(u_reshaped, dtype=np.float32)
        u_reshaped = torch.tensor(u_reshaped[:,::downsampling_factor])
       

        
        predictions = model(u_reshaped).detach().numpy()
    

        y_pred_tmp = np.squeeze(np.asarray(predictions, dtype=np.float32))
        gain[jj], phase[jj] = Get_gain_phase(y_pred_tmp, A, (h+1)*freq)
        
        frequencies[jj] = freq
        jj += 1
 
    return (gain, phase, frequencies )

def plot(model, my_input_shape, path, timeHistorySizeOfU = 100, downsampling_factor = 3):

    
    hf = h5py.File('data/Kornilov_Haeringer_all.h5', 'r')
    run = {}

    # CHECK IF IT'S SAME!!!
    # Generate Impulse Response
    impulse = np.zeros((2*my_input_shape)) # Here maybe
    impulse[my_input_shape-1] = 0.1
    
    impulse_reshaped = np.lib.stride_tricks.sliding_window_view(impulse, my_input_shape)
    impulse_reshaped = np.array(impulse_reshaped, dtype=np.float32)
    impulse_reshaped = torch.tensor(impulse_reshaped)
    
    # Predict
    prediction = model(impulse_reshaped)
    prediction = prediction.detach().numpy()
    
    # Plot
    time = np.linspace(0, downsampling_factor*1e-4*len(prediction), len(prediction))
    IR_Q_ref = np.array(hf.get('IR_Q'))  /(downsampling_factor*100*100)
    IR_time_ref = np.array(hf.get('IR_time'))
    #mse_IR = torch.nn.MSELoss(prediction, IR_Q_ref)

    plt.figure(figsize=(12, 8))
    plt.plot(time,prediction, label='Impulse prediction', color='red')
    plt.plot(IR_time_ref,IR_Q_ref, label='Impulse reference', color='blue')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.legend()
    plt.savefig(f'figures/{path}/IRQ.png')


    # Check how model performs on harmonics
    # load harmonic data
    Harm_A150_f100_Q = np.array(hf.get('Harm_A150_f100_Q'))
    Harm_A100_f100_Q = np.array(hf.get('Harm_A100_f100_Q'))
    Harm_A050_f100_Q = np.array(hf.get('Harm_A050_f100_Q'))
    Harm_A025_f100_Q = np.array(hf.get('Harm_A025_f100_Q'))
    Harm_A0025_f100_Q = np.array(hf.get('Harm_A0025_f100_Q'))

    Harm_A150_f100_Q = Harm_A150_f100_Q[0::100]
    Harm_A100_f100_Q = Harm_A100_f100_Q[0::100]
    Harm_A050_f100_Q = Harm_A050_f100_Q[0::100]
    Harm_A025_f100_Q = Harm_A025_f100_Q[0::100]
    Harm_A0025_f100_Q = Harm_A0025_f100_Q[0::100]



    t = np.linspace(0, 1e-4*len(Harm_A150_f100_Q), len(Harm_A150_f100_Q))
    u =  np.sin(2 * np.pi * 100 * t)
    fake_velocity = np.zeros(timeHistorySizeOfU)
    u = np.concatenate((fake_velocity,u))

    u_reshaped = np.lib.stride_tricks.sliding_window_view(u, timeHistorySizeOfU)
    u_reshaped = u_reshaped[:,::downsampling_factor]
    u_reshaped = np.array(u_reshaped, dtype=np.float32)
    u_reshaped = torch.tensor(u_reshaped)

    prediction_A150 = model(1.5*u_reshaped).detach().numpy()
    prediction_A100 = model(u_reshaped).detach().numpy()
    prediction_A050 = model(0.5*u_reshaped).detach().numpy()
    prediction_A025 = model(0.25*u_reshaped).detach().numpy()
    prediction_A0025 = model(0.025*u_reshaped).detach().numpy()

    errors = {}
    # plot the prediction vs the harmonic data
    plt.figure(figsize=(12, 8))
    plt.plot(Harm_A150_f100_Q, label='Harmonic Data', color='blue', alpha=0.5)
    plt.plot(prediction_A150, label='Prediction', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('A = 150%')
    plt.legend()
    plt.savefig(f'figures/{path}/harmonic_A150.png')
    errors['150'] = compute_harmonics_error(prediction_A150[:-1,:], Harm_A150_f100_Q, 150 / 100)


    plt.figure(figsize=(12, 8))
    plt.plot(Harm_A100_f100_Q, label='Harmonic Data', color='blue', alpha=0.5)
    plt.plot(prediction_A100, label='Prediction', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('A = 100%')
    plt.legend()
    plt.savefig(f'figures/{path}/harmonic_A100.png')
    errors['100'] = compute_harmonics_error(prediction_A100[:-1,:], Harm_A100_f100_Q, 100 / 100)


    plt.figure(figsize=(12, 8))
    plt.plot(Harm_A050_f100_Q, label='Harmonic Data', color='blue', alpha=0.5)
    plt.plot(prediction_A050, label='Prediction', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('A = 50%')
    plt.legend()
    plt.savefig(f'figures/{path}/harmonic_A050.png')
    errors['050'] = compute_harmonics_error(prediction_A050[:-1,:], Harm_A050_f100_Q, 50 / 100)

    plt.figure(figsize=(12, 8))
    plt.plot(Harm_A025_f100_Q, label='Harmonic Data', color='blue', alpha=0.5)
    plt.plot(prediction_A025, label='Prediction', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('A = 25%')
    plt.legend()
    plt.savefig(f'figures/{path}/harmonic_A025.png')
    errors['025'] = compute_harmonics_error(prediction_A025[:-1,:], Harm_A025_f100_Q, 25 / 100)

    plt.figure(figsize=(12, 8))
    plt.plot(Harm_A0025_f100_Q, label='Harmonic Data', color='blue', alpha=0.5)
    plt.plot(prediction_A0025, label='Prediction', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('A = 2.5%')
    plt.legend()
    plt.savefig(f'figures/{path}/harmonic_A0025.png')
    errors['0025'] = compute_harmonics_error(prediction_A0025[:-1,:], Harm_A0025_f100_Q, 2.5 / 100)

    amplitudes = ['150', '100', '050', '025', '0025']
    
    for amp in amplitudes:
        mse, mse_rel, mse_harm = errors[amp]
        run[f'mse_A{amp}'] = mse.item()
        run[f'mse_rel_A{amp}'] = mse_rel.item()
        run[f'mse_harm_A{amp}'] = mse_harm.item()

#######################################################################################################
# FDF freq
    
    Avalues = [0.1, 0.5, 1]
    gain_A10_h0, phase_A10_h0,  frequencies_A10 = calculate_FDF(Avalues[0],0, model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A50_h0, phase_A50_h0, frequencies_A50_h0  = calculate_FDF(Avalues[1],0,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A100_h0, phase_A100_h0, frequencies_A100_h0  = calculate_FDF(Avalues[2],0,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)

    gain_A10_h1, phase_A10_h1,  frequencies_A10_h1 = calculate_FDF(Avalues[0],1,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A50_h1, phase_A50_h1, frequencies_A50_h1  = calculate_FDF(Avalues[1],1,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A100_h1, phase_A100_h1, frequencies_A100_h1  = calculate_FDF(Avalues[2],1,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)

    gain_A10_h2, phase_A10_h2,  frequencies_A10_h2 = calculate_FDF(Avalues[0],2,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A50_h2, phase_A50_h2, frequencies_A50_h2  = calculate_FDF(Avalues[1],2,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A100_h2, phase_A100_h2, frequencies_A100_h2  = calculate_FDF(Avalues[2],2,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)

    gain_A10_h3, phase_A10_h3,  frequencies_A10_h3 = calculate_FDF(Avalues[0],3, model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A50_h3, phase_A50_h3, frequencies_A50_h3  = calculate_FDF(Avalues[1],3,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)
    gain_A100_h3, phase_A100_h3, frequencies_A100_h3  = calculate_FDF(Avalues[2],3,model=model, timeHistorySizeOfU=timeHistorySizeOfU, downsampling_factor=downsampling_factor)


    cfd_freq = [50,100,150,200,250]
    cfd_gain10 = [1.176,1.392,1.279,0.886,0.45]
    cfd_phase10 = [-0.639,-1.6731,-2.847,2.236,0.997]
    cfd_gain50 = [1.14,1.24,1.067,0.767,0.4229]
    cfd_phase50 = [-0.667,-1.69,-2.8,2.311,1.0347]

    # Plots

    fig, axs = plt.subplots(2, 3,figsize=(15,10))
    axs[0, 0].plot(frequencies_A10, gain_A10_h0, label='Gain H0',color='blue')
    axs[0, 0].plot(frequencies_A10, gain_A10_h1, label='Gain H1',color='blue')
    axs[0, 0].plot(frequencies_A10, gain_A10_h2, label='Gain H2',color='blue')
    axs[0, 0].plot(frequencies_A10, gain_A10_h3, label='Gain H3',color='blue')
    axs[0, 0].plot(cfd_freq,cfd_gain10,'o',label='CFD',color='red')
    axs[0, 0].set_title('A = 10%')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Gain')
    axs[0, 0].grid(True)
    # ylim
    axs[0, 0].set_ylim(0.0, 1.7)

    axs[0, 1].plot(frequencies_A50_h0, gain_A50_h0, label='Gain H0',color='blue')
    axs[0, 1].plot(frequencies_A50_h1, gain_A50_h1, label='Gain H1',color='blue')
    axs[0, 1].plot(frequencies_A50_h2, gain_A50_h2, label='Gain H2',color='blue')
    axs[0, 1].plot(frequencies_A50_h3, gain_A50_h3, label='Gain H3',color='blue')
    axs[0, 1].plot(cfd_freq,cfd_gain50,'o',label='CFD',color='red')
    axs[0, 1].set_title('A = 50%')
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Gain')
    axs[0, 1].grid(True)
    # ylim
    axs[0, 1].set_ylim(0.0, 1.7)

    axs[0, 2].plot(frequencies_A100_h0, gain_A100_h0, label='Gain H0',color='blue')
    axs[0, 2].plot(frequencies_A100_h1, gain_A100_h1, label='Gain H1',color='blue')
    axs[0, 2].plot(frequencies_A100_h2, gain_A100_h2, label='Gain H2',color='blue')
    axs[0, 2].plot(frequencies_A100_h3, gain_A100_h3, label='Gain H3',color='blue')
    axs[0, 2].set_title('A = 100%')
    axs[0, 2].set_xlabel('Frequency [Hz]')
    axs[0, 2].set_ylabel('Gain')
    axs[0, 2].grid(True)
    axs[0, 1].set_ylim(0.0, 1.7)

    axs[1, 0].plot(frequencies_A10, phase_A10_h0, label='Phase H0',color='blue')
    axs[1, 0].plot(cfd_freq,cfd_phase10,'o',label='CFD',color='red')
    axs[1, 0].set_title('A = 10%')
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('Phase')
    axs[1, 0].grid(True)
    # ylim

    axs[1, 1].plot(frequencies_A50_h0, phase_A50_h0, label='Phase H0',color='blue')
    axs[1, 1].plot(cfd_freq,cfd_phase50,'o',label='CFD',color='red')
    axs[1, 1].set_title('A = 50%')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Phase')
    axs[1, 1].grid(True)
    # ylim

    axs[1, 2].plot(frequencies_A100_h0, phase_A100_h0, label='Phase H0',color='blue')
    axs[1, 2].plot(cfd_freq,cfd_phase10,'o',label='CFD',color='red')
    axs[1, 2].set_title('A = 100%')
    axs[1, 2].set_xlabel('Frequency [Hz]')
    axs[1, 2].set_ylabel('Phase')
    axs[1, 2].grid(True)
    # ylim

    plt.savefig(f'figures/{path}/gain_phase_graphs.png')

    return run

#############################################################################################################################
