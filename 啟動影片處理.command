#!/bin/sh
cd -- "$(dirname -- "$0")"
sh start-web.sh
vpf_status=$?
if [ "$vpf_status" -ne 0 ]; then
    printf '\n啟動未完成，請檢查上方訊息。按 Enter 關閉。\n'
    read -r vpf_reply
fi
exit "$vpf_status"
