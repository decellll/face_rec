document.addEventListener('DOMContentLoaded', function() {
    // Fungsi untuk mengupdate bar emosi
    function updateEmotionBar(emotionId, value) {
        const bar = document.getElementById(emotionId);
        if (bar) {
            bar.style.width = `${value}%`;
        }
    }

    // Fungsi untuk mensimulasikan update data emosi (contoh)
    function simulateEmotionUpdates() {
        const emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
        
        setInterval(() => {
            emotions.forEach(emotion => {
                const randomValue = Math.floor(Math.random() * 100);
                updateEmotionBar(`${emotion}-bar`, randomValue);
            });
        }, 1000);
    }

    // Mulai simulasi (hapus ini ketika mengintegrasikan dengan data sebenarnya)
    simulateEmotionUpdates();
});
