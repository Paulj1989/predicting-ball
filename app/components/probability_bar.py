# app/components/probability_bar.py


def create_probability_bar(prob, color="#43494D"):
    """Create a visual probability bar with percentage display"""
    percentage = f"{prob * 100:.1f}%"
    bar_width = prob * 100

    return f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="flex-grow: 1; background-color: #F7F9FA; border-radius: 3px; overflow: hidden;">
            <div style="width: {bar_width}%; background-color: {color}; height: 20px;"></div>
        </div>
        <span style="min-width: 50px; font-weight: 500; font-family: 'Poppins', sans-serif;">{percentage}</span>
    </div>
    """
