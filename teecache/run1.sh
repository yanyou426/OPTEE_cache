rm -f result1.txt

# for i in {0..4}
# do
echo -e "----ATTACK ROUND----\n" >> result1.txt
start=$(date +%s.%N);
sudo LD_LIBRARY_PATH=/home/pi/tee/aes ./spy >> result1.txt
dur=$(echo "$(date +%s.%N) - $start" | bc);
echo -e "DURATION $dur" >> result1.txt
# echo -e "----ATTACK ROUND SUCCESS----\n" >> result.txt
# echo -e "\e[1;32m----ATTACK ROUND $i SUCCESS----\e[0m"
# done

echo -e "ATTACK FINISH!"
