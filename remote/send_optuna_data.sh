# Code to send the optimization data
echo "Sending the optuna optimization data to remote ..."
scp -J r0877229@st.cs.kuleuven.be -r ./optuna r0877229@ninove.student.cs.kuleuven.be:~/Documents/remote/