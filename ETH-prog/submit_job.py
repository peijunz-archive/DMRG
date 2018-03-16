import subprocess

for w in [0.2, 0.5, 1, 2, 4, 8, 16]:
    qsub_command = """qsub -v WINDOW={} job.pbs""".format(w)

    print(qsub_command)# Uncomment this line when testing to view the qsub command

    # Comment the following 3 lines when testing to prevent jobs from being submitted
    exit_status = subprocess.call(qsub_command, shell=True)
    if exit_status is 1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(qsub_command))
print("Done submitting jobs!")
