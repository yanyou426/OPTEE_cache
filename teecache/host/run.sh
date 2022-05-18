rm -f result_tee.txt

#for i in {0..4}
#do
echo -e "----ATTACK ROUND----\n" >> result_tee.txt
start=$(date +%s.%N);
sudo ./TEEencrypt >> result_tee.txt
dur=$(echo "$(date +%s.%N) - $start" | bc);
echo -e "DURATION $dur" >> result_tee.txt
#echo -e "----ATTACK ROUND SUCCESS----\n" >> result_tee.txt
#echo -e "\e[1;32m----ATTACK ROUND $i SUCCESS----\e[0m"
#done

echo -e "ATTACK TIME FINISH!"
