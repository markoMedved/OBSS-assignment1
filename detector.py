import scipy.signal as signal
import numpy as np
import wfdb
import sys

def haar_filter(x,fs):
   #compute B1 and B2
    B1 = int(0.025 * fs)  # 9 for fs = 360
    B2 = int(0.06 * fs)   # 21 for fs = 360

    #define c
    c = 2 * (B2 - B1) / (2 * B1 + 1)

    # range for h
    n = np.arange(-B2, B2 + 1)

    #set values of h like described in the paper
    h = np.zeros_like(n, dtype=float)
    h[np.abs(n) <= B1] = c
    h[(np.abs(n) > B1) & (np.abs(n) <= B2)] = -1

    #return the convolution of x and input signal
    return signal.convolve(x,h,mode='same')


def get_candidates(y,x_bsl, fs, c1 = 0.55):
    x2 = np.zeros(len(x_bsl))
    #fill x2 by the formula from paper
    x2[1:len(x_bsl)-1] = np.array( [2 * x_bsl[n] - x_bsl[n+1] - x_bsl[n-1] for n in range(1, len(x_bsl)-1)] )
    
    #calculate s
    s = np.array([y[n]*(x_bsl[n] + c1*x2[n]) for n in range(len(y))])
    
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


#improvements - calculating mean time between heart beats - if heart beat to close - cancel it and take the higher one 


def detect_beats(s,x_bsl,candidates, fs, T=0.1, beta1=0.5, beta2=0.5, taus=[0.08,  0.12, 0.2, 0.25, 0.35], omega_treshold = 0.1):
    #current s in the window
    s_current = []
    #window size
    seconds_window = 10
    samples_window = fs*seconds_window
    #recent detected number
    recent_detected_num = 5
    #detected beats
    detected = []

    #IMPROVEMENT TAKE CHECK MEAN TIME BETWEEN HEART BEATS
    #for start take 0.5 seconds - 120 beats per minute
    mean_time_secs = 0.5
    mean_time_samples_start = mean_time_secs*fs
    mean_time_samples = mean_time_samples_start


    #for every candidate
    for i in candidates:
        s_current = []
        for j in range(i-samples_window, i):
            #skip if out of range
            if j < 0:
                continue
            s_current.append(abs(s[j]))

        #calculate W1, get 5th largest value in the last 10 seconds
        W1 = T
        if len(s_current) >= 5:
            W1 = sorted(s_current)[-5] +T

            
        #calculate W2
        W2 = beta1
        det_len = len(detected)
        #only if there is at least 5 detected beat 
        if det_len > 5:
            recent_det = detected[det_len - recent_detected_num: det_len]

            mew = np.mean([recent_det[i] - recent_det[i+1] for i in range(3)])



            if recent_det[4] - recent_det[3] < 0.7*mew and det_len > 6:
                tmp = recent_det
                recent_det = [detected[-6]]
                recent_det.extend(tmp)
                recent_det = recent_det[:5]

                Ie = sum([taus[i] * (recent_det[i] - recent_det[i+1]) for i in range(4)])
                W2 = beta1 + beta2 * abs((i - recent_det[4])/(2*Ie) - round((i - recent_det[4])/(2*Ie)))
                print(W2)
        
            else:
                Ie = sum([taus[i] * (recent_det[i] - recent_det[i+1]) for i in range(4)])
                W2 = beta1 + beta2 * abs((i - recent_det[4])/Ie - round((i - recent_det[4])/Ie))
            
        
        treshold = W1*W2

        #variation ratio
        omega = 1/2

        if i - int(0.1*fs) > 0:
            tmp_list=  [x_bsl[j] for j in range(i - int(0.1*fs), i + int(0.1*fs))]
            u1 = max(tmp_list) - min(tmp_list)
            u2 = sum([abs(x_bsl[j] - x_bsl[j-1]) for j in range(i - int(0.1*fs)+1, i + int(0.1*fs) +1)])
            omega = u1/u2

        

      
        if abs(s[i]) > treshold:  

            if len(detected)>=2:
                samples_between = detected[-1] - detected[-2]
                #calculate new mean
                mean_time_samples = mean_time_samples * (len(detected)-1)/len(detected) + samples_between/len(detected)
                #so that it doesn't go out of control
                mean_time_samples = min([mean_time_samples_start, mean_time_samples])



            #IMPROVEMENT
            #if next beat is way to close to the first one - don't append it
            if omega > omega_treshold:
                if len(detected) < 1:
                    detected.append(i)
                elif (i - detected[-1]) > 0.7*mean_time_samples:
                    detected.append(i)
                    
                # if too close take the one with the larger s
                else:
                 
                    if s[i] > detected[-1]:
           
                        detected[-1] = i




    return detected



def detector(filename_no_extension):
    sig = wfdb.rdsamp(filename_no_extension)
    x_both = sig[0][:].flatten()
    x = x_both[0::2]

    #read sampling frequency
    fs = wfdb.rdsamp(filename_no_extension)[1]['fs']
    #read annotation file
    record_name = filename_no_extension
    annotation = wfdb.rdann(record_name, 'atr')

    #get rid of baseline
    butter_filter = signal.butter(5, 2, 'high', fs=fs, output='sos')
    x_bsl = signal.sosfilt(butter_filter, x)
    x_bsl[::-1] = signal.sosfilt(butter_filter, x_bsl[::-1])

    #delay
    delay = 1
    x_bsl[:-delay] = x_bsl[delay:]

    #apply haar filter
    y = haar_filter(x_bsl,fs)

    #get candidates
    candidates,s = get_candidates(y,x_bsl,fs)

    #detect beats
    detected = detect_beats(s,x_bsl,candidates, fs, T=1, beta2=0.01, beta1=0.1)

    symbols = ["N"] * len(detected)


    wfdb.io.wrann(filename_no_extension, "qrs", np.array(detected), symbols)

    return

if __name__ == "__main__":
    detector(sys.argv[1])