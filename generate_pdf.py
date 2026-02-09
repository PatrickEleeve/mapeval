"""
Generate PDF from code and output - complete code display
"""
import subprocess
import sys
import textwrap
from pathlib import Path

def generate_pdf():
    # Read the code file
    with open('data_analyst.py', 'r') as f:
        code = f.read()
    
    # Run the script and capture output
    result = subprocess.run([sys.executable, 'data_analyst.py'], 
                          capture_output=True, text=True, encoding='utf-8')
    
    # Combine stdout and stderr to ensure all output is captured
    output = result.stdout
    if result.stderr:
        output += '\n' + result.stderr
    
    # Verify output is not empty
    if not output or not output.strip():
        print("⚠ Warning: No output captured from data_analyst.py")
        output = "[No output generated]"
    else:
        print(f"✓ Captured {len(output)} characters of output")
        print(f"  Output preview (first 200 chars): {output[:200]}...")
    
    # Create PDF using matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    pdf_path = 'code_sample.pdf'
    code_lines = code.split('\n')
    total_lines = len(code_lines)
    
    with PdfPages(pdf_path) as pdf:
        # Optimized settings for maximum content
        fontsize = 7.2
        line_height = 0.0145  # Slightly tighter spacing
        lines_per_page = 60  # More lines per page
        
        # Page 1: First half of code
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.04, 0.03, 0.92, 0.94])  # Use more of the page
        ax.axis('off')
        
        ax.text(0.5, 0.99, 'Financial Data Analysis', 
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                ha='center', va='top')
        
        y_pos = 0.96
        # Split roughly in half, but ensure we get all code
        mid_point = total_lines // 2
        
        for i, line in enumerate(code_lines[:mid_point]):
            if y_pos < 0.02:
                break
            # Handle long lines by wrapping
            if len(line) > 115:
                wrapped = textwrap.wrap(line, width=115)
                for wline in wrapped:
                    ax.text(0.01, y_pos, wline, transform=ax.transAxes, 
                           fontsize=fontsize, fontfamily='monospace', 
                           va='top', ha='left')
                    y_pos -= line_height
                    if y_pos < 0.02:
                        break
            else:
                ax.text(0.01, y_pos, line, transform=ax.transAxes, 
                       fontsize=fontsize, fontfamily='monospace', 
                       va='top', ha='left')
                y_pos -= line_height
        
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Page 2: Remaining code
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.04, 0.03, 0.92, 0.94])
        ax.axis('off')
        
        ax.text(0.5, 0.99, 'Financial Data Analysis', 
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                ha='center', va='top')
        
        y_pos = 0.96
        
        # Show ALL remaining code
        remaining_code_lines = code_lines[mid_point:]
        
        for line in remaining_code_lines:
            if y_pos < 0.02:
                break
            # Handle long lines
            if len(line) > 115:
                wrapped = textwrap.wrap(line, width=115)
                for wline in wrapped:
                    ax.text(0.01, y_pos, wline, transform=ax.transAxes, 
                           fontsize=fontsize, fontfamily='monospace', 
                           va='top', ha='left')
                    y_pos -= line_height
                    if y_pos < 0.02:
                        break
            else:
                ax.text(0.01, y_pos, line, transform=ax.transAxes, 
                       fontsize=fontsize, fontfamily='monospace', 
                       va='top', ha='left')
                y_pos -= line_height
        
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Page 3: Output only
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.04, 0.03, 0.92, 0.94])
        ax.axis('off')
        
        ax.text(0.5, 0.99, 'PROGRAM OUTPUT', 
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                ha='center', va='top')
        
        y_pos = 0.96
        
        # Output content - use slightly smaller font
        output_lines = output.split('\n')
        output_fontsize = 6.5
        
        # Track how many output lines are actually displayed
        displayed_output_lines = 0
        
        for line in output_lines:
            if y_pos < 0.02:
                break
            if len(line) > 115:
                wrapped = textwrap.wrap(line, width=115)
                for wline in wrapped:
                    ax.text(0.01, y_pos, wline, transform=ax.transAxes, 
                           fontsize=output_fontsize, fontfamily='monospace', 
                           va='top', ha='left')
                    y_pos -= line_height
                    if y_pos < 0.02:
                        break
            else:
                ax.text(0.01, y_pos, line, transform=ax.transAxes, 
                       fontsize=output_fontsize, fontfamily='monospace', 
                       va='top', ha='left')
                y_pos -= line_height
            displayed_output_lines += 1
        
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Page 4: Image only
        import os
        img_path = 'financial_analysis.png'
        if os.path.exists(img_path):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.04, 0.03, 0.92, 0.94])
            ax.axis('off')
            
            try:
                from PIL import Image
                import numpy as np
                
                pil_img = Image.open(img_path)
                max_width = 1200
                if pil_img.width > max_width:
                    ratio = max_width / pil_img.width
                    new_height = int(pil_img.height * ratio)
                    pil_img = pil_img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                img_array = np.array(pil_img)
                
                page_aspect = 8.5 / 11
                img_aspect = pil_img.width / pil_img.height
                
                img_height = 0.75
                img_width = img_height * img_aspect / page_aspect
                
                if img_width > 0.88:
                    img_width = 0.88
                    img_height = img_width * page_aspect / img_aspect
                
                img_x_center = 0.5
                img_left = img_x_center - img_width / 2
                img_right = img_x_center + img_width / 2
                img_y_pos = 0.95
                img_bottom = img_y_pos - img_height
                
                ax.imshow(img_array, extent=[img_left, img_right, img_bottom, img_y_pos], 
                         aspect='auto', interpolation='bilinear', zorder=1, origin='upper')
                
            except ImportError:
                from matplotlib import image as mpimg
                img = mpimg.imread(img_path)
                img_aspect = img.shape[1] / img.shape[0]
                page_aspect = 8.5 / 11
                
                img_height = 0.75
                img_width = img_height * img_aspect / page_aspect
                if img_width > 0.88:
                    img_width = 0.88
                    img_height = img_width * page_aspect / img_aspect
                
                img_x_center = 0.5
                img_y_pos = 0.95
                img_bottom = img_y_pos - img_height
                
                ax.imshow(img, extent=[img_x_center - img_width/2, img_x_center + img_width/2,
                                      img_bottom, img_y_pos], 
                         aspect='auto', interpolation='bilinear', zorder=1, origin='upper')
            except Exception as e:
                ax.text(0.5, 0.5, f'[Image could not be loaded: {str(e)}]', 
                       transform=ax.transAxes, fontsize=10, ha='center', va='center',
                       style='italic', color='gray')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()
    
    # Verify code completeness
    code_in_pdf_lines = mid_point + len(remaining_code_lines)
    total_output_lines = len(output.split('\n'))
    
    print(f"✓ PDF generated: {pdf_path}")
    print(f"  Total code lines in file: {total_lines}")
    print(f"  Code lines in PDF: {code_in_pdf_lines}")
    print(f"  Total output lines: {total_output_lines}")
    print(f"  Output lines displayed: {displayed_output_lines}")
    print(f"  File size: {Path(pdf_path).stat().st_size / 1024:.1f} KB")
    
    if code_in_pdf_lines == total_lines:
        print("  ✓ All code lines included in PDF")
    else:
        print(f"  ⚠ Warning: {total_lines - code_in_pdf_lines} lines may be missing")
    
    if displayed_output_lines < total_output_lines:
        print(f"  ⚠ Warning: {total_output_lines - displayed_output_lines} output lines may be truncated")
    else:
        print("  ✓ All output lines included in PDF")
    
    return pdf_path

if __name__ == '__main__':
    generate_pdf()
