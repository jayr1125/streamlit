import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from pycaret.time_series import *

st.set_page_config(layout="wide")

# Display FLNT logo
image = Image.open(r"flnt logo.png")
st.sidebar.image(image,
                 width=160)

# Display file uploader (adding space beneath the FLNT logo)
st.sidebar.write("")
st.sidebar.write("")
data1 = st.sidebar.file_uploader("Upload File 1 Here", type=["csv", "xls", "xlsx"])
data2 = st.sidebar.file_uploader("Upload File 2 Here", type=["csv", "xls", "xlsx"])

st.sidebar.write("---")

# Check for errors during upload
try:
    # Read dataset file uploader
    if data1 is not None:
        if data1.name.endswith(".csv"):
            data_df1 = pd.read_csv(data1)
        else:
            data_df1 = pd.read_excel(data1)

    if data2 is not None:
        if data2.name.endswith(".csv"):
            data_df2 = pd.read_csv(data2)
        else:
            data_df2 = pd.read_excel(data2)

    # For choosing features and targets
    data_df1_types = data_df1.dtypes.to_dict()
    data_df2_types = data_df2.dtypes.to_dict()

    # Choosing features and target for file 1
    targets1 = []
    for key, val in data_df1_types.items():
        if val != object:
            targets1.append(key)

    chosen_target1 = st.sidebar.selectbox(f"Choose a target for {data1.name}",
                                          targets1)
    features1 = list(data_df1_types.keys())
    features1.remove(chosen_target1)
    chosen_features1 = st.sidebar.multiselect(f"Choose features for {data1.name}",
                                              features1)

    st.sidebar.write("---")

    new_cols1 = chosen_features1.copy()
    new_cols1.append(chosen_target1)

    data_df1 = data_df1[new_cols1]

    # Choosing features and target for file 2
    targets2 = []
    for key, val in data_df2_types.items():
        if val != object:
            targets2.append(key)

    chosen_target2 = st.sidebar.selectbox(f"Choose a target for {data2.name}",
                                          targets2)
    features2 = list(data_df2_types.keys())
    features2.remove(chosen_target2)
    chosen_features2 = st.sidebar.multiselect(f"Choose features for {data2.name}",
                                              features2)

    new_cols2 = chosen_features2.copy()
    new_cols2.append(chosen_target2)

    data_df2 = data_df2[new_cols2]

    # Preprocess data for experiment setup
    data_df1_series = data_df1.copy()
    data_df2_series = data_df2.copy()

    # For descriptive stats
    data_df1_cols = data_df1.columns
    data_df2_cols = data_df2.columns
    data_df1_shape = data_df1.shape
    data_df2_shape = data_df2.shape
    data_df1_missing = data_df1[data_df1_cols[1]].isnull().sum()
    data_df2_missing = data_df2[data_df2_cols[1]].isnull().sum()

    data_df1_series[data_df1_series.columns[0]] = pd.to_datetime((data_df1_series[data_df1_series.columns[0]]),
                                                                 format="%Y-%m-%d")
    data_df2_series[data_df2_series.columns[0]] = pd.to_datetime((data_df2_series[data_df2_series.columns[0]]),
                                                                 format="%Y-%m-%d")

    data_df1_series.set_index(data_df1_series[data_df1_series.columns[0]],
                              inplace=True)
    data_df1_series.drop(data_df1_series.columns[0],
                         axis=1,
                         inplace=True)

    data_df2_series[data_df2_series.columns[0]] = pd.to_datetime((data_df2_series[data_df2_series.columns[0]]),
                                                                 format="%Y-%m-%d")
    data_df2_series[data_df2_series.columns[0]] = pd.to_datetime((data_df2_series[data_df2_series.columns[0]]),
                                                                 format="%Y-%m-%d")

    data_df2_series.set_index(data_df2_series[data_df2_series.columns[0]],
                              inplace=True)
    data_df2_series.drop(data_df2_series.columns[0],
                         axis=1,
                         inplace=True)

    # Create tabs for plots and statistics
    plot_tab, stat_tab = st.tabs(["Plots", "Statistics"])

    with plot_tab:
        # Make 2 columns
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(f"Plots for {data1.name}")

            # Data 1 plot
            fig1 = go.Figure()
            fig1.add_trace(go.Line(name=data1.name,
                                   x=data_df1_series.index,
                                   y=data_df1_series[data_df1_cols[1]]))
            fig1.update_xaxes(gridcolor='grey')
            fig1.update_yaxes(gridcolor='grey')
            fig1.update_layout(colorway=["#7ee3c9"],
                               xaxis_title=data_df1_cols[0],
                               yaxis_title=data_df1_cols[1],
                               title=f"{data_df1_cols[0]} vs. {data_df1_cols[1]}")

            st.plotly_chart(fig1,
                            use_container_width=True)

            # Setup experiment
            s1 = setup(data_df1_series,
                       fh=3,
                       fold=5,
                       session_id=123)
            seasonal_data1 = s1.seasonality_present
            white_noise_data1 = s1.white_noise

            stats1 = check_stats()
            mask_stats1 = (stats1['Property'] == 'Mean') | \
                          (stats1['Property'] == 'Median') | \
                          (stats1['Property'] == 'Standard Deviation') | \
                          (stats1['Property'] == 'Stationarity')
            relevant_stats1 = stats1[mask_stats1]
            relevant_stats1.drop(['Test', 'Test Name', 'Data', 'Setting'], axis=1, inplace=True)

            mean_data1 = round(relevant_stats1.iloc[0]['Value'], 2)
            median_data1 = round(relevant_stats1.iloc[1]['Value'], 2)
            std_data1 = round(relevant_stats1.iloc[2]['Value'], 2)
            stationarity_data1 = relevant_stats1.iloc[3]['Value']

            best_data1 = compare_models()
            results1 = pull()
            results1.Model.tolist()
            model_name1 = results1.iloc[0]['Model']

            data1_slider = st.slider(f"Forecast Horizon for {data1.name}", 1, 10, 5)
            plot_data1 = plot_model(best_data1,
                                    plot='forecast',
                                    data_kwargs={'fh': data1_slider},
                                    return_data=True)

            x_data1 = plot_data1['data'][0]['x']
            y_data1 = plot_data1['data'][0]['y']
            y_upper1 = plot_data1['predictions'][0]['upper'].values
            y_lower1 = plot_data1['predictions'][0]['lower'].values

            # Forecast 1 plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Data",
                x=data_df1_series.index,
                y=data_df1_series[data_df1_cols[1]]
            ))

            fig2.add_trace(go.Scatter(
                name='Prediction',
                x=x_data1,
                y=y_data1,
                # mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))

            fig2.add_trace(go.Scatter(
                name='Upper Bound',
                x=x_data1,
                y=y_upper1,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))

            fig2.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_data1,
                y=y_lower1,
                marker=dict(color="#70B0E0"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

            fig2.update_xaxes(gridcolor='grey')
            fig2.update_yaxes(gridcolor='grey')
            fig2.update_layout(xaxis_title=data_df1_cols[0],
                               yaxis_title=data_df1_cols[1],
                               title=f"{data1.name} Forecast using {model_name1}",
                               hovermode="x",
                               colorway=["#7ee3c9"])

            st.plotly_chart(fig2,
                            use_container_width=True)

        with c2:
            st.subheader(f"Plots for {data2.name}")

            fig3 = go.Figure()
            fig3.add_trace(go.Line(name=data2.name,
                                   x=data_df2_series.index,
                                   y=data_df2_series[data_df1_cols[1]]))
            fig3.update_xaxes(gridcolor='grey')
            fig3.update_yaxes(gridcolor='grey')
            fig3.update_layout(colorway=["#7ee3c9"],
                               xaxis_title=data_df2_cols[0],
                               yaxis_title=data_df2_cols[1],
                               title=f"{data_df2_cols[0]} vs. {data_df2_cols[1]}")

            st.plotly_chart(fig3,
                            use_container_width=True)

            # Setup experiment
            s2 = setup(data_df2_series,
                       fh=3,
                       fold=5,
                       session_id=123)

            seasonal_data2 = s2.seasonality_present
            white_noise_data2 = s2.white_noise

            stats2 = check_stats()
            mask_stats2 = (stats2['Property'] == 'Mean') | \
                          (stats2['Property'] == 'Median') | \
                          (stats2['Property'] == 'Standard Deviation') | \
                          (stats2['Property'] == 'Stationarity')
            relevant_stats2 = stats2[mask_stats2]
            relevant_stats2.drop(['Test', 'Test Name', 'Data', 'Setting'],
                                 axis=1,
                                 inplace=True)

            mean_data2 = round(relevant_stats2.iloc[0]['Value'], 2)
            median_data2 = round(relevant_stats2.iloc[1]['Value'], 2)
            std_data2 = round(relevant_stats2.iloc[2]['Value'], 2)
            stationarity_data2 = relevant_stats2.iloc[3]['Value']

            best_data2 = compare_models()
            results2 = pull()
            results2.Model.tolist()
            model_name2 = results2.iloc[0]['Model']

            data2_slider = st.slider(f"Forecast Horizon for {data2.name}", 1, 10, 5)

            plot_data2 = plot_model(best_data2,
                                    plot='forecast',
                                    data_kwargs={'fh': data2_slider},
                                    return_data=True)

            x_data2 = plot_data2['data'][0]['x']
            y_data2 = plot_data2['data'][0]['y']
            y_upper2 = plot_data2['predictions'][0]['upper'].values
            y_lower2 = plot_data2['predictions'][0]['lower'].values

            fig4 = go.Figure()

            fig4.add_trace(go.Scatter(
                name="Data",
                x=data_df2_series.index,
                y=data_df2_series[data_df1_cols[1]]
            ))

            fig4.add_trace(go.Scatter(
                name='Prediction',
                x=x_data2,
                y=y_data2,
                # mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))

            fig4.add_trace(go.Scatter(
                name='Upper Bound',
                x=x_data2,
                y=y_upper2,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))

            fig4.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_data2,
                y=y_lower2,
                marker=dict(color="#70B0E0"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

            fig4.update_xaxes(gridcolor='grey')
            fig4.update_yaxes(gridcolor='grey')
            fig4.update_layout(xaxis_title=data_df2_cols[0],
                               yaxis_title=data_df2_cols[1],
                               title=f"{data2.name} Forecast using {model_name2}",
                               hovermode="x",
                               colorway=["#7ee3c9"])

            st.plotly_chart(fig4,
                            use_container_width=True)

        # Show correlation coefficient of the uploaded files (must be the same target name)
        corr = round(data_df1.corrwith(data_df2)[data_df1_cols[1]], 2)

        # Correlation Plot
        fig5 = go.Figure()
        fig5.add_trace(go.Line(name=data1.name,
                               x=data_df1_series.index,
                               y=data_df1_series[data_df1_cols[1]]))
        fig5.add_trace(go.Line(name=data2.name,
                               x=data_df2_series.index,
                               y=data_df2_series[data_df1_cols[1]]))
        fig5.update_xaxes(gridcolor='grey')
        fig5.update_yaxes(gridcolor='grey')
        fig5.update_layout(colorway=["#7ee3c9", "#70B0E0"],
                           xaxis_title=data_df1_cols[0],
                           yaxis_title=data_df1_cols[1],
                           title="Data Correlation")

        st.plotly_chart(fig5,
                        use_container_width=True)

        st.metric(f"Correlation of {data1.name} and {data2.name}", str(int((corr*100))) + "%")

    with stat_tab:

        st.header("Descriptive Statistics")
        st.write("---")
        c3, c4 = st.columns(2)

        with c3:
            st.subheader(f"{data1.name}")
            # Show descriptive statistics for file 1
            st.metric("No. of Variables", data_df1_shape[1])
            st.metric("No. of Observations", data_df1_shape[0])
            st.metric("No. of Missing Values", data_df1_missing)
            st.metric("Mean", mean_data1)
            st.metric("Median", median_data1)
            st.metric("Standard Deviation", std_data1)
            st.metric("Seasonality", seasonal_data1)
            st.metric("Stationarity", stationarity_data1)
            st.metric("White Noise", white_noise_data1)

        with c4:
            st.subheader(f"{data2.name}")
            # Show descriptive statistics for file 2
            st.metric("No. of Variables", data_df2_shape[1])
            st.metric("No. of Observations", data_df2_shape[0])
            st.metric("No. of Missing Values", data_df2_missing)
            st.metric("Mean", mean_data2)
            st.metric("Median", median_data2)
            st.metric("Standard Deviation", std_data2)
            st.metric("Seasonality", seasonal_data2)
            st.metric("Stationarity", stationarity_data2)
            st.metric("White Noise", white_noise_data2)

except NameError:
    pass

print("Done Rendering Application!")
