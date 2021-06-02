#!/bin/sh

while :
do
    check_python_process_by_pid=$( ps ax | grep 3211 )
    now=$(date)
    
    if  [[ $check_python_process_by_pid == *"related_articles.py"* ]]
    then
        echo $now ": ML Service is running"
        sleep 60
    else
        echo $now ": ML Service not running, sending email alert"
        sudo ssmtp nbardwell910@gmail.com -F ML-Alert < mail.txt
        sleep 300
    fi
done