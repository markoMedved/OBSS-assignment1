import scipy.signal as signal
import numpy as np
import wfdb

def haar_filter(x,fs):
    #constants
    B1 = int(0.025*fs)
    B2 = int(0.06*fs)
    c = 2 * (B2 - B1)/(2*B1 +1)
    #signals
    y1 = np.zeros(len(x))
    y2= np.zeros(len(x))
    y = np.zeros(len(x))
    tmp= np.zeros(B2)
    x_padded = np.concatenate((tmp,x,tmp))
    #go through samples
    h = np.zeros(2*(B1+B2))
    tmp = np.zeros(2*(B1+B2))
    tmp[:B2] = c
    tmp[B2:B1+B2] = -1
    h = np.concatenate((tmp[::-1],tmp))

    #return signal.convolve(x,h,mode='same')

    for i in range(len(y)):
        #plus B2 in x[] is because of padding
        y1[i] = (c+1)*(x_padded[i+B1 + B2] - x_padded[i-B1 +B2]) + y1[i-1]
        y2[i] = -(x_padded[i+B2 + B2] - x_padded[i-B2 + B2] + y2[i-1])
        y[i] = y1[i] + y2[i]
        #y[i] = -sum(x_padded[i-B1+B2:i+B1+B2+1]) + (c+1) * sum(x_padded[i-B2+B2:i+B2+B2+1])
    return y


def get_candidates(y,x_no_baseline, fs, c1 = 0.55):
    x2 = np.zeros(len(x_no_baseline))
    #fill x2 by the formula from paper
    x2[1:len(x_no_baseline)-1] = np.array( [2 * x_no_baseline[n] - x_no_baseline[n+1] - x_no_baseline[n-1] for n in range(1, len(x_no_baseline)-1)] )
    #calculate s
    s = np.array([y[n]*(x_no_baseline[n] + c1*x2[n]) for n in range(len(y))])
    #tmp just to find the correct range of checking
    tmp = int(0.2*fs)
    #find candidates
    candidates =[]
    for k in range(tmp, len(s)-tmp):
        for n in range(-tmp, tmp):
            if np.abs(s[k]) < np.abs(s[k+n]):
                break
            elif n == tmp-1:
                candidates.append(k)

    return candidates, s


def detect_beats(s,x_no_baseline,candidates, fs, T=0.1, beta1=0.5, beta2=0.5, taus=[0.08,  0.12, 0.2, 0.25, 0.35], omega_treshold = 0.1):
    
    #current s in the window
    s_current = []
    #window size
    seconds_window = 10
    samples_window = fs*seconds_window

    #recent detected number
    recent_detected_num = 5

    #detected beats
    detected = []

    #for every candidate
    for i in candidates:
        s_current = []
        for j in range(i-samples_window, i):
            #skip if out of range
            if j < 0:
                continue
            s_current.append(abs(s[j]))

        #calculate W1
        W1 = T
        if len(s_current) > 5:
            W1 = sorted(s_current)[-5] +T
            

        #calculate W2
        W2 = beta1
        det_len = len(detected)
        #only if there is at least 5 detected beats
        if det_len > 5:
            recent_det = detected[det_len - recent_detected_num: det_len]

            mew = np.mean([recent_det[i] - recent_det[i+1] for i in range(3)])


            
            if recent_det[-2] - recent_det[-3] < 0.7*mew and det_len > 6:
              
                tmp = recent_det

                recent_det = [detected[-6]]
                recent_det.extend(tmp[:-1])

            recent_det.append(i)
                

            Ie = sum([taus[i] * (recent_det[i] - recent_det[i+1]) for i in range(5)])

            
            W2 = beta1 + beta2 * abs((i - recent_det[0])/Ie - round((i - recent_det[0])/Ie))
            
        treshold = W1*W2


        omega = 1/2
        if i - int(0.1*fs) > 0:
            tmp_list=  [x_no_baseline[j] for j in range(i - int(0.1*fs), i + int(0.1*fs))]
            u1 = max(tmp_list) - min(tmp_list)
            u2 = sum([abs(x_no_baseline[j] - x_no_baseline[j-1]) for j in range(i - int(0.1*fs), i + int(0.1*fs))])
            omega = u1/u2

      
        if s[i] > treshold:
            
            if omega > omega_treshold:
                detected.append(i)
            else:
                print(omega, omega_treshold, i)

        #else :
            #print(s[i], treshold, i)

    return detected


def detector(filename_no_extension):
    
    sig = wfdb.rdsamp(filename_no_extension)
    x_both = sig[0][:].flatten()
    x = x_both[0::2]

    #read sampling frequency
    fs = wfdb.rdsamp(filename_no_extension)[1]['fs']

    #read annotation file
    annotation = wfdb.rdann(filename_no_extension, 'atr')
    beat_peaks = annotation.sample[:]

    #get rid of baseline
    butter_filter = signal.butter(5, 2, 'high', fs=fs, output='sos')
    x_no_baseline = signal.sosfilt(butter_filter, x)
    x_no_baseline[::-1] = signal.sosfilt(butter_filter, x_no_baseline[::-1])

    #delay
    delay = 1
    x_no_baseline[:-delay] = x_no_baseline[delay:]

    #apply haar filter
    y = haar_filter(x_no_baseline,fs)

    candidates,s = get_candidates(y,x_no_baseline,fs)

    detected = detect_beats(s,x_no_baseline,candidates, fs, T=5, beta2=0.01, beta1=0.2)

    time_tolerance = 0.015
    tolerance = round(time_tolerance*fs)

    not_detected = []
    for i in beat_peaks:
        for j in range(i-tolerance, i+tolerance+1):
            if j in detected:
                break
        if j == i+tolerance:
            not_detected.append(i)

    falsely_detected = []
    for i in detected:
        for j in range(i-tolerance, i+tolerance+1):
            if j in beat_peaks:
                break
        if j == i+tolerance:
            falsely_detected.append(i)

    SE = (len(detected) - len(falsely_detected))/len(beat_peaks)
    PP = (len(detected) - len(falsely_detected))/len(detected) 
    print('Sensitivity:', SE)
    print('Positive predictivity:', PP)

    with open("annotations.asc", "w") as file:
        for sample in detected:
            file.write(str(sample) + "\n")



def main():
    detector("mit-bih-data/105")


if __name__ == "__main__":
    main()