// Function to update the status bar based on a value (from 0 to 100)
function updateStatusBar(value) {
    const statusBar = document.getElementById('statusBar');
    const statusText = document.getElementById('status-text');

    // Ensure the value is within the range [0, 100]
    const clampedValue = Math.min(Math.max(value, 0), 100);

    // Calculate the color based on the value (Red to Yellow to Green scale)
    const red = Math.floor((100 - clampedValue) * 255 / 50);
    const green = Math.floor(clampedValue * 255 / 50);
    const color = `rgb(${red},${green},0)`;

    // Update the status bar width and color
    statusBar.style.width = `${clampedValue}%`;
    statusBar.style.backgroundColor = color;

    // Update the status text
    statusText.innerText = `${clampedValue}%`;
    statusText.style.color = color;
}

