// Credit Card Approval Prediction System - Frontend Interactions

document.addEventListener('DOMContentLoaded', () => {
    // 1. Batch Table Client-side Search Filter
    const searchInput = document.getElementById('batchSearchInput');
    const table = document.getElementById('batchTable');
    
    if (searchInput && table) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const rows = table.querySelectorAll('tbody tr');
            
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                if (text.includes(query)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }

    // 2. Smooth Form Interaction Feedback
    const predForm = document.getElementById('predictionForm');
    if (predForm) {
        predForm.addEventListener('submit', () => {
            const submitBtn = predForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scoring Profile...';
                submitBtn.disabled = true;
            }
        });
    }

    // 3. Highlight algorithm changes
    const algSelect = document.getElementById('algorithm');
    if (algSelect) {
        algSelect.addEventListener('change', () => {
            if (algSelect.value === 'best') {
                algSelect.classList.add('highlight-select');
            } else {
                algSelect.classList.remove('highlight-select');
            }
        });
    }
});
