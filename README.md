<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>medSeg - Medical Segmentation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }

        h1,
        h2,
        h3 {
            color: #333;
        }

        a {
            color: #0066cc;
            text-decoration: none;
        }

        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            overflow: auto;
        }

        code {
            font-family: monospace;
        }

        ul {
            list-style-type: disc;
            margin-left: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .section {
            margin: 40px 0;
        }

        .note {
            font-style: italic;
            color: #555;
        }

        hr {
            margin: 20px 0;
            border: none;
            border-top: 1px solid #ddd;
        }
    </style>
</head>

<body>
    <div class="container">
        <header>
            <h1>medSeg</h1>
            <p class="note">Medical Segmentation (medSeg) - Analyze and process CT scans for tumor detection and mutation analysis.</p>
        </header>

        <hr />

        <section class="section">
            <h2>📄 Description</h2>
            <p><strong>medSeg</strong> is a Medical Segmentation Tool for analyzing and processing CT images. The application provides functionalities to detect areas containing cancer, predict mutations, and save segmented images for further analysis.</p>
        </section>

        <section class="section">
            <h2>🚀 Getting Started</h2>
            <h3>📌 Requirements</h3>
            <p>Ensure you have Python <strong>3.x</strong> installed with the required dependencies:</p>
            <pre><code>pip install -r requirements.txt</code></pre>
            <h3>🛠️ Dependencies</h3>
            <ul>
                <li><code>PyQt5</code> - GUI library for user interaction.</li>
                <li><code>torch</code>, <code>torchvision</code> - Libraries for machine learning model processing.</li>
                <li><code>cv2</code> (OpenCV) - For image processing and visualization.</li>
                <li><code>PIL</code> - For image manipulation.</li>
                <li><code>ultralytics</code> - YOLO model support for object detection.</li>
                <li><code>numpy</code> - For numerical processing.</li>
            </ul>
        </section>

        <section class="section">
            <h2>📂 Installation & Launch</h2>
            <ol>
                <li>Clone the repository:
                    <pre><code>git clone https://github.com/&lt;your-repository-name&gt;.git
cd medSeg</code></pre>
                </li>
                <li>Install dependencies:
                    <pre><code>pip install -r requirements.txt</code></pre>
                </li>
                <li>Launch the application:
                    <pre><code>python main.py</code></pre>
                </li>
            </ol>
        </section>

        <section class="section">
            <h2>🖥️ Usage Instructions</h2>
            <h3>🔹 Browse for Data Folder</h3>
            <p>Use the <strong>"Browse"</strong> button to select a directory containing raw CT images. Once selected, the images will populate a dropdown menu for analysis.</p>
            <h3>🔹 Process Data</h3>
            <p>Press the <strong>"Process"</strong> button to start processing the selected image folder. Segmented images and masks will be saved in the following directories:</p>
            <ul>
                <li><code>detected/</code> - Contains detection results (tumor detection regions).</li>
                <li><code>segmentated/</code> - Contains segmented tumor masks.</li>
            </ul>
            <h3>🔹 View Results</h3>
            <p>After processing:</p>
            <ul>
                <li>Select an image from the dropdown menu.</li>
                <li>View the raw CT image, segmented tumor masks, and analysis results (EGFR, KRAS mutations) directly in the GUI.</li>
            </ul>
        </section>

        <section class="section">
            <h2>📊 Example Workflow</h2>
            <p>After running the application and processing images:</p>
            <ul>
                <li><strong>Original Images</strong> - Raw CT images are visualized.</li>
                <li><strong>Segmentation Masks</strong> - Saved as <code>mask.png</code>.</li>
                <li><strong>Mutation Analysis Results</strong> - Predictions for EGFR and KRAS mutations are displayed in the GUI.</li>
            </ul>
        </section>

        <section class="section">
            <h2>🛡️ License</h2>
            <p>This software is licensed under the <a href="https://opensource.org/licenses/MIT" target="_blank">MIT License</a>.</p>
            <p>You can use <strong>medSeg</strong> for personal or commercial projects. When integrating it into your project, ensure the license remains included.</p>
        </section>

        <section class="section">
            <h2>🧑‍💻 Developers</h2>
            <p><strong>Author:</strong> Your Name</p>
            <p><strong>Email:</strong> your-email@example.com</p>
            <p><strong>GitHub:</strong> <a href="https://github.com/your-repository-name" target="_blank">https://github.com/your-repository-name</a></p>
            <p>If you have any questions, feature requests, or bug reports, feel free to open an issue in this repository.</p>
        </section>

        <section class="section">
            <h2>🏆 Acknowledgments</h2>
            <ul>
                <li><a href="https://www.riverbankcomputing.com/software/pyqt/" target="_blank">PyQt5</a> - For building interactive user interfaces.</li>
                <li><a href="https://github.com/ultralytics/yolov5" target="_blank">YOLO (ultralytics)</a> - For object detection support.</li>
                <li><a href="https://pytorch.org/vision/stable/index.html" target="_blank">torchvision</a> - For image transformations and pre-trained models.</li>
            </ul>
        </section>
    </div>
</body>

</html>
