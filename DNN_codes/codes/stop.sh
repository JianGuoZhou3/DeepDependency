for pid in $(ps -ef | awk '/SdA/ {print $2}'); do kill -9 $pid; done
