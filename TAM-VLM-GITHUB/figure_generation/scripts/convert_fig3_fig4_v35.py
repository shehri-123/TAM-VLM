#!/usr/bin/env python3

"""
TAM-VLM v35
Convert Fig.3 and Fig.4 source diagrams
to IEEE/Q1 ready PDFs
"""


from PIL import Image
import matplotlib.pyplot as plt
import os


from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
INPUT_DIR = os.path.join(
    BASE,
    "figures",
    "final_v35"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "figures"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



def convert_png_to_pdf(
        input_file,
        output_file
):

    img = Image.open(input_file)

    img = img.convert("RGB")


    w,h = img.size


    fig_width = 7.16

    fig_height = (
        h / w
    ) * fig_width


    fig = plt.figure(
        figsize=(
            fig_width,
            fig_height
        )
    )


    ax = fig.add_axes(
        [0,0,1,1]
    )


    ax.imshow(img)

    ax.axis("off")


    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0
    )


    plt.close()


    print("Created:", output_file)



# ===============================
# Fig 3
# ===============================


fig3_input = os.path.join(
    INPUT_DIR,
    "Final 2 fig.png"
)


fig3_output = os.path.join(
    OUTPUT_DIR,
    "fig3_benchmark_protocol_v35.pdf"
)



convert_png_to_pdf(
    fig3_input,
    fig3_output
)



# ===============================
# Fig 4
# ===============================


fig4_input = os.path.join(
    INPUT_DIR,
    "Final fig.png"
)


fig4_output = os.path.join(
    OUTPUT_DIR,
    "fig4_trigger_examples_v35.pdf"
)



convert_png_to_pdf(
    fig4_input,
    fig4_output
)



print("\n================================")
print("Fig.3 + Fig.4 completed")
print("================================")
