
# Define the commands you want to run
commands=(
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 0-100"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 100-200"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 200-300"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 300-400"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 400-500"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 500-600"
    "source /apps/anaconda3/bin/activate && conda activate sbert && python topic_inference.py -batch_id_range 600-700"
)

# Run each command in a separate terminal window
for cmd in "${commands[@]}"; do
    gnome-terminal -- bash -c "$cmd; exec bash"
done


# python topic_inference.py -batch_id_range 0-100
# python topic_inference.py -batch_id_range 100-200
# python topic_inference.py -batch_id_range 200-300
# python topic_inference.py -batch_id_range 300-400
# python topic_inference.py -batch_id_range 400-500
# python topic_inference.py -batch_id_range 500-600
# python topic_inference.py -batch_id_range 600-700