import sys
import os


def unite_pdfs(in_path,out_path):
    fnames = []
    for fname in os.listdir(in_path):
        if fname.endswith(".pdf"):
            fnames.append(fname)
        
    fnames.sort(key=lambda x:int("".join(filter(str.isdigit,x))))

    fnames = [os.path.join(in_path,fname) for fname in fnames]

    os.system("pdfunite {} {}".format(" ".join(fnames),out_path))

if __name__=="__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    unite_pdfs(in_path,out_path)