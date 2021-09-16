import sys
import os
dest = "data/mesh_files"
def genmesh():
    import subprocess
    if not os.path.exists(dest):
        os.makedirs(dest)
    else:
        print("You have mesh files")
        return
    fname = 'icosphere_{}.pkl'
    for i in range(8):
        url = 'http://island.me.berkeley.edu/ugscnn/mesh_files/' + fname.format(i)
        command = ["wget", "--no-check-certificate", "-P", dest, url]
        try:
            download_state = subprocess.call(command)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    genmesh()
