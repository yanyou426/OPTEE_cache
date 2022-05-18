rm -f result.txt

for i in {0..4}
do
    echo -e "----ATTACK ROUND $i----\n" >> result.txt
    start=$(date +%s.%N);
    sudo LD_LIBRARY_PATH=/home/pi/tee/aes ./elimi >> result.txt
    dur=$(echo "$(date +%s.%N) - $start" | bc);
    echo -e "DURATION $dur" >> result.txt
    echo -e "----ATTACK ROUND $i SUCCESS----\n" >> result.txt
    echo -e "\e[1;32m----ATTACK ROUND $i SUCCESS----\e[0m"
done

echo -e "\n----ATTACK 5 TIMES FINISH!----\n"
