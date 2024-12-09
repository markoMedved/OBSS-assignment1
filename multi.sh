
## Script for running C program and evaluation of the algorithm.
##########

rm eval1.txt
rm eval2.txt

FILES=*.dat

# Compile detector
#gcc -o myDetector -O myDetector.c -lm -lwfdb

# For all record files run detector
for f in $FILES
do
  f=$(basename $f)
  f=${f%.*}

  echo  $f 
     python detector.py $f

  bxb -r $f -a atr qrs -l eval1.txt eval2.txt
 done
 
# Calculate aggregate statistics
sumstats eval1.txt eval2.txt >results_improved.txt


# dat - signal recordings from the database
# atr - reference annotations from the database
# qrs - annotations of implemented algorithm
# results.txt - final statistics for reporting

# If your implementation is in Matalab, then comment out compiling and running
# the C program and just use the part for evaluation (bxb and sumstats).

# Other explanations of how to run programs and convert records and annotation 
# files are described on the web-classrom under certain laboratory sessions.
