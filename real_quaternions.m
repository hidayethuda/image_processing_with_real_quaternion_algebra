
% Class for image processing with the help of quaternion matrix theory.
% The 'real_quaternions' class provides a framework for various
% image processing tasks, including image representation, image
% reconstruction, and watermarking with the singular value decomposition(SVD)
% algorithm derived in quaternion matrix algebra.

classdef real_quaternions

    properties

        r_part;
        i_part;
        j_part;
        k_part;

    end

    methods(Static, Access=public)

        function obj=image_to_quaternion(I)

            % image_to_quaternion: Convert an input image into a matrix of real quaternions.
            %
            % Inputs:
            %   I - A 3-channel image (typically in RGB format).
            %
            % Outputs:
            %   obj - An object that belongs to the class of real quaternions, representing
            %   the input image.
            % Example Usage:
            %   obj=real_quaternions.image_to_quaternion(image);

            I =im2double(I);
            [R,G,B]=imsplit(I);
            obj=real_quaternions(zeros(size(R)),R,G,B);

        end

        function obj=complex_to_quaternion(C)

            % complex_to_quaternion: Convert an adjoint matrix to the corresponding quaternion matrix.
            % Example Usage:
            %  obj=real_quaternions.complex_to_quaternion(Adjoint_matrix);

            M=size(C,1)/2;
            N=size(C,2)/2;
            obj.r_part=real(C(1:M,1:N));
            obj.i_part=imag(C(1:M,1:N));
            obj.j_part=real(C(1:M,N+1:2*N));
            obj.k_part=imag(C(1:M,N+1:2*N));
            obj=real_quaternions(obj.r_part,obj.i_part,obj.j_part,obj.k_part);
            
        end

    end

    methods (Access=public)

        function obj=real_quaternions(varargin)
            if nargin == 4
                obj.r_part=varargin{1};
                obj.i_part=varargin{2};
                obj.j_part=varargin{3};
                obj.k_part=varargin{4};
            else
                error('Invalid arguments: quaternions must have 4 components...')
            end

        end

        function obj = quaternion_product(obj1, obj2)

            obj.r_part = obj1.r_part * obj2.r_part - obj1.i_part * obj2.i_part - obj1.j_part * obj2.j_part - obj1.k_part * obj2.k_part;
            obj.i_part = obj1.r_part * obj2.i_part + obj1.i_part * obj2.r_part + obj1.j_part * obj2.k_part - obj1.k_part * obj2.j_part;
            obj.j_part = obj1.r_part * obj2.j_part - obj1.i_part * obj2.k_part + obj1.j_part * obj2.r_part + obj1.k_part * obj2.i_part;
            obj.k_part = obj1.r_part * obj2.k_part + obj1.i_part * obj2.j_part - obj1.j_part * obj2.i_part + obj1.k_part * obj2.r_part;
            obj=real_quaternions(obj.r_part,obj.i_part,obj.j_part,obj.k_part);

        end

        function obj = quaternion_addition(obj1, obj2)

            obj.r_part = obj1.r_part + obj2.r_part;
            obj.i_part = obj1.i_part + obj2.i_part;
            obj.j_part = obj1.j_part + obj2.j_part;
            obj.k_part = obj1.k_part + obj2.k_part;
            obj=real_quaternions(obj.r_part,obj.i_part,obj.j_part,obj.k_part);

        end

        function obj=quaternion_scalarproduct(lambda,obj1)
            
            obj=real_quaternions(lambda*obj1.r_part,lambda*obj1.i_part,lambda*obj1.j_part,lambda*obj1.k_part);

        end

        function adjoint_matrix=quaternion_to_complex(obj)

            % quaternion_to_complex: Convert a quaternion matrix to its corresponding adjoint matrix.
            %
            % Inputs:
            %   obj - An object of the class of real quaternions.
            %
            % Outputs:
            %   C - A complex matrix representing the adjoint matrix of the quaternion.

            X1 = complex(obj.r_part, obj.i_part);
            X2 = complex(obj.j_part, obj.k_part);
            adjoint_matrix = [X1 X2; -conj(X2) conj(X1)];

        end

        function obj=quaternion_hermitianconjugate(obj1)

            C=quaternion_to_complex(obj1);
            C=ctranspose(C);
            obj=real_quaternions.complex_to_quaternion(C);

        end

        function I=quaternion_to_image(obj)

            % quaternion_to_image: Convert a quaternion matrix to an image.
            %
            % Inputs:
            %   obj - An object of the class of real quaternions.
            %
            % Outputs:
            %   I - A 3-channel image (RGB format) constructed from the quaternion matrix.

            R=obj.i_part;
            G=obj.j_part;
            B=obj.k_part;
            I=cat(3,R,G,B);
            I=im2double(I);

        end

        function [U,S,V] = quaternion_svd(obj)

            % quaternion_svd: Compute the Singular Value Decomposition (SVD) of a quaternion matrix.
            %
            % Inputs:
            %   obj - An object of the class of real quaternions.
            %
            % Outputs:
            %   U - The left singular vectors of the quaternion matrix, represented as a quaternion matrix.
            %   S - The diagonal matrix of singular values, represented as a quaternion matrix with real parts.
            %   V - The right singular vectors of the quaternion matrix, represented as a quaternion matrix.
            %
            % Reference:
            %   Soo-Chang Pei, Ja-Han Chang, and Jian-Jiun Ding,
            %   "Quaternion matrix singular value decomposition and its applications for color image processing,"
            %   Proceedings 2003 International Conference on Image Processing (Cat. No.03CH37429), Barcelona, Spain, 2003,
            %   pp. I-805, doi: 10.1109/ICIP.2003.1247084

            M=size(quaternion_to_complex(obj),1);
            N=size(quaternion_to_complex(obj),2);
            [U_c,S_c,V_c] = svd(quaternion_to_complex(obj));
            zero_element=zeros(size(S_c)/2);
            diagonal_elements = sort(diag(S_c),'descend');
            odd_index_elements = diagonal_elements(1:2:end);
            real_part = diag(odd_index_elements);
            [rowsreal_part, colsreal_part] = size(real_part);
            real_part1=zero_element;
            real_part1(1:rowsreal_part, 1:colsreal_part) = real_part;
            S=real_quaternions(real_part1,zero_element,zero_element,zero_element);
            U_c(:, 2:2:end) = [];
            V_c(:, 2:2:end) = [];
            U=real_quaternions(real(U_c(1:M/2,1:M/2)),imag(U_c(1:M/2,1:M/2)),real(-conj(U_c((M/2)+1:M,1:M/2))),imag(-conj(U_c((M/2)+1:M,1:M/2))));
            V=real_quaternions(real(V_c(1:N/2,1:N/2)),imag(V_c(1:N/2,1:N/2)),real(-conj(V_c((N/2)+1:N,1:N/2))),imag(-conj(V_c((N/2)+1:N,1:N/2))));

        end

        function quaternion_eigenimage(obj,h)

            % quaternion_eigenimage: Calculate and display the h-th eigenimage
            % of the image expressed by the quaternion matrix.
            %
            % Inputs:
            %   obj - An object of the class of real quaternions.
            %   h   - The index of the eigenimage to compute and display.
            %
            % Outputs:
            %   None. This function displays the eigenimage in a new figure.

            [U,~,V]=quaternion_svd(obj);
            U_h_column=real_quaternions(U.r_part(:,h),U.i_part(:,h),U.j_part(:,h),U.k_part(:,h));
            V_h_column=real_quaternions(V.r_part(:,h),V.i_part(:,h),V.j_part(:,h),V.k_part(:,h));
            quaternion_eimage=quaternion_product(U_h_column,quaternion_hermitianconjugate(V_h_column));
            newimage=quaternion_to_image(quaternion_eimage);
            absolute_image = abs(newimage);
            min_value = min(absolute_image(:));
            max_value = max(absolute_image(:));
            normalized_image = (absolute_image - min_value) / (max_value - min_value);
            figure
            imshow(normalized_image);
            title(sprintf('%d st eigenimage', h));

        end

        function  quaternion_singularvalues(obj)

            %quaternion_singularvalues: It shows the change of singular values
            % of the image expressed with a quaternion matrix on a log10 scale.

            [~,S,~]=quaternion_svd(obj);
            y=zeros(rank(S.r_part),1);
            for j=1:rank(S.r_part)
                y(j)=log10(S.r_part(j,j));
            end     
            figure
            plot(y,'r','lineWidth',2);
            title('change of singular values of the test image')
            ylabel('log10 scale')
            
        end

        function quaternion_image_reconstruction(obj,k)

            % quaternion_image_reconstruction: Reconstruct and display an image
            % from its quaternion SVD approximation.
            % Reference:
            %   Soo-Chang Pei, Ja-Han Chang, and Jian-Jiun Ding,
            %   "Quaternion matrix singular value decomposition and its applications for color image processing,"
            %   Proceedings 2003 International Conference on Image Processing (Cat. No.03CH37429), Barcelona, Spain, 2003,
            %   pp. I-805, doi: 10.1109/ICIP.2003.1247084

            [U,S,V]=quaternion_svd(obj);
            U=real_quaternions(U.r_part(1:end,1:k),U.i_part(1:end,1:k),U.j_part(1:end,1:k),U.k_part(1:end,1:k));
            S=real_quaternions(S.r_part(1:k,1:k),zeros(k),zeros(k),zeros(k));
            V=real_quaternions(V.r_part(1:end,1:k),V.i_part(1:end,1:k),V.j_part(1:end,1:k),V.k_part(1:end,1:k));           
            recon_image=quaternion_product(quaternion_product(U,S),quaternion_hermitianconjugate(V));         
            I_c=quaternion_to_image(recon_image);
            I_o=quaternion_to_image(obj);
            PSNR=psnr(I_c,I_o);
            MSE=immse(I_c,I_o);
            figure
            imshow(I_c);
            title_text = compose('Compressed Image-PSNR %.4f and MSE %.5f', PSNR, MSE);
            title(title_text);

        end

        function  quaternion_watermarking(obj1,obj2,alpha)

            % quaternion_watermarking: Embed a watermark into an image using quaternion matrix decomposition.
            %
            % Inputs:
            %   obj1 - An object of the class of real quaternions representing the original image.
            %   obj2 - An object of the class of real quaternions representing the watermark image.
            %   alpha - The scaling factor to control the strength of the watermark.
            %
            % Outputs:
            %   None. This function displays the original image, watermark image, and watermarked image with PSNR and MSE metrics.
            % Reference:
            %  Abd El-Samie, F.E., "An efficient singular value decomposition algorithm for digital audio watermarking,"
            %  Int J Speech Technol 12, 27–45 (2009). https://doi.org/10.1007/s10772-009-9056-2

            original_image=quaternion_to_image(obj1);
            water_mark_image=quaternion_to_image(obj2);
            [U_c,S_c,V_c] = quaternion_svd(obj1);
            obj2=real_quaternions.image_to_quaternion(imresize(water_mark_image, [size(obj1.r_part,1) size(obj1.r_part,2)]));
            [~,S_w,~] = quaternion_svd(quaternion_addition(S_c,quaternion_scalarproduct(alpha,obj2)));
            quaternion_water_marked_image=quaternion_product(quaternion_product(U_c,S_w),quaternion_hermitianconjugate(V_c));   
            water_marked_image=quaternion_to_image(quaternion_water_marked_image);
            PSNR=psnr(original_image,water_marked_image);
            MSE=immse(original_image,water_marked_image);
            title_text = compose('PSNR %.4f and MSE %d', PSNR, MSE);    
            figure
            subplot(1,3,1)
            imshow(original_image);
            title('Original image')
            subplot(1,3,2)
            imshow(water_mark_image)
            title('Water mark image')
            subplot(1,3,3)
            imshow(water_marked_image)
            title('Water marked image')
            sgtitle(title_text)
            
        end

    end 

end 

