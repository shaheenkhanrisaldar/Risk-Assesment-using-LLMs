import pandas as pd
from fpdf import FPDF

# Load data from CSV
def load_data(filepath):
    return pd.read_csv(filepath)

# Generate text summary for each stock entry, ensuring ASCII compatibility
def generate_summary(df):
    summaries = []
    for _, row in df.iterrows():
        try:
            summary = (f"{row['Name']} ({row['Ticker']}) belongs to {row['Sub-Sector']} sector with a market cap of INR {row['Market Cap']:.2f} Crores. "
                       f"The closing price is INR {row['Close Price']}. It has a PE Ratio of {row['PE Ratio']:.2f}, ROCE of {row['ROCE']:.2f}%, "
                       f"Net Profit Margin of {row['Net Profit Margin']:.2f}%, Return on Equity of {row['Return on Equity']:.2f}%, "
                       f"Return on Assets of {row['Return on Assets']:.2f}%, EBITDA Margin of {row['EBITDA Margin']:.2f}%, "
                       f"Return on Investment of {row['Return on Investment']:.2f}%, Quick Ratio of {row['Quick Ratio']:.2f}, "
                       f"Current Ratio of {row['Current Ratio']:.2f}, Net Income to Liabilities Ratio of {row['Net Income / Liabilities']:.2f}, "
                       f"Debt to Equity Ratio of {row['Debt to Equity']:.2f}, Dividend Yield of {row['Dividend Yield']:.2f}%, "
                       f"Sector PE of {row['Sector PE']:.2f}, EBITDA of INR {row['EBITDA']}, PBIT of INR {row['PBIT']}, "
                       f"Net Income of INR {row['Net Income']}, and Earnings Per Share of INR {row['Earnings Per Share']:.2f}.")
            # Encode to ASCII and decode back to remove non-ASCII characters
            summary = summary.encode('ascii', errors='ignore').decode()
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing summary for {row['Ticker']}: {e}")
    return summaries

# Create a PDF from the summaries
def create_pdf(summaries, filename='Summary_Report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for summary in summaries:
        pdf.add_page()  # Ensures each summary starts on a new page
        pdf.multi_cell(0, 10, txt=summary)
    pdf.output(filename)
    return filename

# Main function to load data, process it and generate PDF
def main(filepath):
    data = load_data(filepath)
    summaries = generate_summary(data)
    pdf_filename = create_pdf(summaries)
    print(f"PDF generated successfully at: {pdf_filename}")

if __name__ == "__main__":
    main(r'C:\Users\LENOVO\Downloads\Nifty500Data.csv')  # Use a raw string for the file path
